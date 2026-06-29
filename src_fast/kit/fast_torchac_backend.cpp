#include <torch/extension.h>

#include <string>

namespace py = pybind11;

using cdf_t = uint16_t;

struct CdfPtr {
    const cdf_t* data;
    const int n_sym;
    const int lp;
};

static CdfPtr get_cdf_ptr(const torch::Tensor& cdf) {
    TORCH_CHECK(!cdf.is_cuda(), "cdf must be on CPU");
    TORCH_CHECK(cdf.dim() == 2, "cdf must have shape [N, Lp]");
    TORCH_CHECK(cdf.is_contiguous(), "cdf must be contiguous");
    TORCH_CHECK(cdf.scalar_type() == torch::kInt16, "cdf must be int16");
    return CdfPtr{
        reinterpret_cast<const cdf_t*>(cdf.data_ptr<int16_t>()),
        static_cast<int>(cdf.size(0)),
        static_cast<int>(cdf.size(1)),
    };
}

static void check_symbols(const torch::Tensor& sym, int n_sym) {
    TORCH_CHECK(!sym.is_cuda(), "symbols must be on CPU");
    TORCH_CHECK(sym.dim() == 1, "symbols must have shape [N]");
    TORCH_CHECK(sym.is_contiguous(), "symbols must be contiguous");
    TORCH_CHECK(sym.scalar_type() == torch::kInt16, "symbols must be int16");
    TORCH_CHECK(sym.size(0) == n_sym, "symbol count does not match cdf");
}

struct OutBits {
    std::string out;
    uint8_t cache = 0;
    uint8_t count = 0;

    explicit OutBits(int n_sym) {
        out.reserve(std::max(16, n_sym));
    }

    inline void append(int bit) {
        cache = static_cast<uint8_t>((cache << 1) | bit);
        count += 1;
        if (count == 8) {
            out.append(reinterpret_cast<const char*>(&cache), 1);
            count = 0;
            cache = 0;
        }
    }

    inline void append_bit_and_pending(int bit, uint64_t& pending_bits) {
        append(bit);
        while (pending_bits > 0) {
            append(!bit);
            pending_bits -= 1;
        }
    }

    inline void flush() {
        if (count > 0) {
            while (count != 0) {
                append(0);
            }
        }
    }
};

struct InBits {
    const std::string& in;
    uint8_t cache = 0;
    uint8_t cached_bits = 0;
    size_t ptr = 0;

    explicit InBits(const std::string& data) : in(data) {}

    inline void get(uint32_t& value) {
        if (cached_bits == 0) {
            if (ptr == in.size()) {
                value <<= 1;
                return;
            }
            cache = static_cast<uint8_t>(in[ptr]);
            ptr += 1;
            cached_bits = 8;
        }
        value <<= 1;
        value |= (cache >> (cached_bits - 1)) & 1;
        cached_bits -= 1;
    }

    inline void initialize(uint32_t& value) {
        for (int i = 0; i < 32; ++i) {
            get(value);
        }
    }
};

static std::string encode_core(const CdfPtr& cdf_ptr, const torch::Tensor& sym) {
    OutBits out_cache(cdf_ptr.n_sym);

    uint32_t low = 0;
    uint32_t high = 0xFFFFFFFFU;
    uint64_t pending_bits = 0;

    const cdf_t* cdf = cdf_ptr.data;
    const int n_sym = cdf_ptr.n_sym;
    const int lp = cdf_ptr.lp;
    const int max_symbol = lp - 2;
    const int16_t* sym_ptr = sym.data_ptr<int16_t>();

    for (int i = 0; i < n_sym; ++i) {
        const int16_t sym_i = sym_ptr[i];
        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
        const int offset = i * lp;
        const uint32_t c_low = cdf[offset + sym_i];
        const uint32_t c_high = sym_i == max_symbol ? 0x10000U : cdf[offset + sym_i + 1];

        high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> 16);
        low = low + ((span * static_cast<uint64_t>(c_low)) >> 16);

        while (true) {
            if (high < 0x80000000U) {
                out_cache.append_bit_and_pending(0, pending_bits);
                low <<= 1;
                high = (high << 1) | 1;
            } else if (low >= 0x80000000U) {
                out_cache.append_bit_and_pending(1, pending_bits);
                low <<= 1;
                high = (high << 1) | 1;
            } else if (low >= 0x40000000U && high < 0xC0000000U) {
                pending_bits += 1;
                low = (low << 1) & 0x7FFFFFFFU;
                high = (high << 1) | 0x80000001U;
            } else {
                break;
            }
        }
    }

    pending_bits += 1;
    if (low < 0x40000000U) {
        out_cache.append_bit_and_pending(0, pending_bits);
    } else {
        out_cache.append_bit_and_pending(1, pending_bits);
    }
    out_cache.flush();
    return out_cache.out;
}

static inline cdf_t find_symbol(const cdf_t* cdf, uint16_t count, int max_symbol, int offset) {
    cdf_t left = 0;
    cdf_t right = static_cast<cdf_t>(max_symbol + 1);
    while (left + 1 < right) {
        const cdf_t mid = static_cast<cdf_t>((left + right) >> 1);
        const cdf_t value = cdf[offset + mid];
        if (value <= count) {
            left = mid;
        } else {
            right = mid;
        }
    }
    return left;
}

static torch::Tensor decode_core(const CdfPtr& cdf_ptr, const std::string& in) {
    const cdf_t* cdf = cdf_ptr.data;
    const int n_sym = cdf_ptr.n_sym;
    const int lp = cdf_ptr.lp;
    const int max_symbol = lp - 2;

    auto out = torch::empty({n_sym}, torch::TensorOptions().dtype(torch::kInt16));
    int16_t* out_ptr = out.data_ptr<int16_t>();

    uint32_t low = 0;
    uint32_t high = 0xFFFFFFFFU;
    uint32_t value = 0;

    InBits in_cache(in);
    in_cache.initialize(value);

    for (int i = 0; i < n_sym; ++i) {
        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
        const uint16_t count = ((static_cast<uint64_t>(value) - static_cast<uint64_t>(low) + 1) * 0x10000U - 1) / span;
        const int offset = i * lp;
        const cdf_t sym_i = find_symbol(cdf, count, max_symbol, offset);
        out_ptr[i] = static_cast<int16_t>(sym_i);

        if (i == n_sym - 1) {
            break;
        }

        const uint32_t c_low = cdf[offset + sym_i];
        const uint32_t c_high = sym_i == max_symbol ? 0x10000U : cdf[offset + sym_i + 1];

        high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> 16);
        low = low + ((span * static_cast<uint64_t>(c_low)) >> 16);

        while (true) {
            if (low >= 0x80000000U || high < 0x80000000U) {
                low <<= 1;
                high = (high << 1) | 1;
                in_cache.get(value);
            } else if (low >= 0x40000000U && high < 0xC0000000U) {
                low = (low << 1) & 0x7FFFFFFFU;
                high = (high << 1) | 0x80000001U;
                value -= 0x40000000U;
                in_cache.get(value);
            } else {
                break;
            }
        }
    }

    return out;
}

static py::bytes encode_cdf_fast(const torch::Tensor& cdf, const torch::Tensor& sym) {
    const auto cdf_ptr = get_cdf_ptr(cdf);
    check_symbols(sym, cdf_ptr.n_sym);
    std::string encoded;
    {
        py::gil_scoped_release release;
        encoded = encode_core(cdf_ptr, sym);
    }
    return py::bytes(encoded);
}

static torch::Tensor decode_cdf_fast(const torch::Tensor& cdf, const std::string& in) {
    const auto cdf_ptr = get_cdf_ptr(cdf);
    torch::Tensor decoded;
    {
        py::gil_scoped_release release;
        decoded = decode_core(cdf_ptr, in);
    }
    return decoded;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_cdf", &encode_cdf_fast, "Encode int16 normalized CDF");
    m.def("decode_cdf", &decode_cdf_fast, "Decode int16 normalized CDF");
}
