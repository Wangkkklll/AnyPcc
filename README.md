<h1 align="center">AnyPcc: Compressing Any Point Cloud with a Single Universal Model </h1>

<p align="center">
    <strong><a href="https://github.com/Wangkkklll">Kangli Wang</a></strong><sup>1</sup>,
    <strong><a href="https://github.com/GongsunBABA">Qianxi Yi</a></strong><sup>1,2</sup>, 
    <strong><a href="https://shihao-homepage.com">Yuqi Ye</a></strong><sup>1</sup>,
    <strong><a href="https://github.com/GongsunBABA">Shihao Li</a></strong><sup>1</sup>,
    <strong><a href="https://gaowei262.github.io/">Wei Gao</a></strong><sup>1,2*</sup><br>
    (<em>* Corresponding author</em>)
</p>

<p align="center">
    <sup>1</sup>SECE, Peking University<br>
    <sup>2</sup>Peng Cheng Laboratory, Shenzhen, China
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2510.20331"><img src="https://img.shields.io/badge/Arxiv-2510.20331-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
  <a href="https://github.com/Wangkkklll/AnyPcc?tab=MIT-1-ov-file"><img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"></a>
  <a href="https://anypcc.github.io/"><img src="https://img.shields.io/badge/Project_Page-AnyPcc-blue.svg" alt="Home Page"></a>
</p>

> **TL;DR:** AnyPcc compress **any source** point cloud with a single universal model.  

## 📣 News
- [25-10-24] 🔥 We initially released the paper and project.

## Todo
- [ ] Release inference code
- [ ] Release training code
- [ ] Release all dataset

## Links
Our work on point cloud or 3DGS compression has also been released. Welcome to check it.
- 🔥 [UniPCGC](https://uni-pcgc.github.io/) [AAAI'25]: A unified point cloud geometry compression. [[`Paper`](https://ojs.aaai.org/index.php/AAAI/article/view/33387)] [[`Arxiv`](https://arxiv.org/abs/2503.18541)] [[`Project`](https://uni-pcgc.github.io/)]
- 🔥 [GausPcc](https://gauspcc.github.io/) [Arxiv'25]: Efficient 3D Gaussian Compression ! [[`Arxiv`](https://arxiv.org/abs/2505.18197)] [[`Project`](https://gauspcc.github.io/)]


## 📌 Introduction

Generalization remains a critical challenge for deep learning-based point cloud geometry compression. We argue this stems from two key limitations: the lack of robust context models and the inefficient handling of out-of-distribution (OOD) data. To address both, we introduce AnyPcc, a universal point cloud compression framework. AnyPcc first employs a Universal Context Model that leverages priors from both spatial and channel-wise grouping to capture robust contextual dependencies. Second, our novel Instance-Adaptive Fine-Tuning (IAFT) strategy tackles OOD data by synergizing explicit and implicit compression paradigms. It fine-tunes a small subset of network weights for each instance and incorporates them into the bitstream, where the marginal bit cost of the weights is dwarfed by the resulting savings in geometry compression. Extensive experiments on a benchmark of 15 diverse datasets confirm that AnyPcc sets a new state-of-the-art in point cloud compression. Our code and datasets will be released to encourage reproducible research.

<div align="center">
<img src="assets/overview.png" width = 75% height = 75%/>
<br>
Ilustration of the proposed framework. 
</div>


## 🔎 Contact
If your have any comments or questions, feel free to contact [kangliwang@stu.pku.edu.cn](kangliwang@stu.pku.edu.cn).

