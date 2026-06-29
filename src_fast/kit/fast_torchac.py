import os

from torch.utils.cpp_extension import load


_SRC = os.path.join(os.path.dirname(__file__), 'fast_torchac_backend.cpp')
_BACKEND = load(
    name='anypcc_fast_torchac_backend',
    sources=[_SRC],
    extra_cflags=['-O3'],
    verbose=os.environ.get('ANYPCC_VERBOSE_BUILD', '0') == '1',
)


def encode_int16_normalized_cdf(cdf_int, sym):
    return _BACKEND.encode_cdf(cdf_int.contiguous(), sym.contiguous())


def decode_int16_normalized_cdf(cdf_int, byte_stream):
    return _BACKEND.decode_cdf(cdf_int.contiguous(), byte_stream)
