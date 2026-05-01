# Sri Harshavardhan Reddy Deverapalli

**GPU Kernel & Performance Engineer | CUDA · CUTLASS · Tensor Cores · HPC**

<p align="left">
  <img src="https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/CUTLASS-111111?style=flat-square&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Tensor%20Cores-111111?style=flat-square&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Nsight%20Compute-76B900?style=flat-square&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/C%2B%2B-00599C?style=flat-square&logo=cplusplus&logoColor=white" />
  <img src="https://img.shields.io/badge/HPC-444444?style=flat-square" />
  <img src="https://img.shields.io/badge/MPI-444444?style=flat-square" />
  <img src="https://img.shields.io/badge/OpenMP-444444?style=flat-square" />
  <img src="https://img.shields.io/badge/CMake-064F8C?style=flat-square&logo=cmake&logoColor=white" />
  <img src="https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
</p>

I build and profile CUDA/CUTLASS kernels for tensor, sparse, and AI workloads. My MS thesis developed a block-sparse tensor contraction pipeline on NVIDIA H100, achieving **3.01× mean / 6.06× max speedup over cuTENSOR 2.5.0 block-sparse API** and reaching **~95% of H100 FP64 Tensor Core peak**.

## Current Focus

- **CUDA kernel profiling studies:** GEMM, WMMA Tensor Core GEMM, reductions, softmax, and FlashAttention-lite
- **GPU performance analysis:** Nsight Compute, Nsight Systems, roofline modeling, occupancy, memory bandwidth, and register-pressure tuning
- **Research:** high-performance block-sparse tensor contractions for quantum many-body simulation on NVIDIA H100

## Selected Work

| Work | Result | Hardware |
|---|---:|---|
| Block-sparse tensor contraction pipeline | 3.01× mean / 6.06× max over cuTENSOR block-sparse API | H100 |
| FP64 Tensor Core tuning | ~95% of H100 FP64 Tensor Core peak | H100 |
| CUDA GEMM profiling study | 73% of cuBLAS | RTX 3050 |
| WMMA Tensor Core GEMM | 82% of cuBLAS | RTX 3050 |
| FlashAttention-lite | 85% of cuDNN SDPA | RTX 3050 |

## Research Note

The H100 block-sparse tensor contraction implementation is private while the related paper is under preparation. Public materials are available through my thesis and SIAM PP26 talk.

## Publications

- *Image compression using quantum wavelet transforms*, SPIE Quantum Computing, Communication, and Simulation V, 2025
- *On-board classification of underwater images using hybrid classical-quantum CNN*, Quantum Machine Intelligence, 2024

## Links

[LinkedIn](https://www.linkedin.com/in/shr-deverapalli) · [Google Scholar](https://scholar.google.com/citations?user=tcn7np0AAAAJ&hl=en) · [Email](mailto:sdevera@ncsu.edu)
