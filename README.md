# Sri Harshavardhan Reddy Deverapalli

**GPU Kernel & Performance Engineer** | MS Computer Science @ NC State (GPA 3.95) | Graduating May 2026 · Open to start dates from June 2026

I build and tune CUDA/CUTLASS kernels for tensor, sparse, and AI workloads on modern GPU architectures. My MS thesis developed a block-sparse tensor contraction pipeline on H100, achieving **3.01× mean speedup** over NVIDIA's cuTENSOR 2.5.0 block-sparse API and reaching **~95% of FP64 Tensor Core peak** (~57.2 of 60 TFLOPS). I explored 3,000+ CUTLASS kernel configurations and built an O(1) rule-based kernel selection framework that replaces exhaustive autotuning.

Currently seeking **GPU kernel engineering** and **HPC** roles.

---

### What I Work On

- **GPU Kernel Engineering** — CUDA, CUTLASS 2.x/3.x, CuTe, inline PTX (WMMA/MMA/GMMA), Tensor Core optimization, shared-memory tiling, warp/warpgroup primitives, SYCL/oneAPI
- **Performance Analysis** — Nsight Compute, Nsight Systems, roofline modeling, occupancy & register-pressure tuning, memory-bandwidth vs compute tradeoff analysis
- **AI/ML Kernels & Inference** — Triton, FlashAttention-style tiled attention, fused softmax/bias/residual, PyTorch C++/CUDA custom ops, mixed-precision (FP16/FP8/INT8), KV cache & paged attention, DDP/FSDP

**Hardware:** H100 (SM90), A100 (SM80), Jetson Xavier, Intel PVC via oneAPI

---
<!--
### AI Kernel Portfolio

> *Standalone GPU kernels for AI/ML workloads — each written from scratch, benchmarked, and profiled.*

| Kernel | What It Demonstrates | Implementation |
|--------|---------------------|----------------|
| **FlashAttention-Lite** | Tiled attention forward pass with online softmax — avoids materializing N×N attention matrix in HBM | CUDA |
| **Fused Softmax** | Numerically stable softmax with warp-shuffle reduction, single-pass | CUDA + Triton |
| **Tiled GEMM** | FP32 shared-memory tiled matrix multiply with bank-conflict-free padding | CUDA |
| **Mixed-Precision GEMM** | FP16 inputs, FP32 accumulation via WMMA Tensor Core intrinsics | CUDA + PTX |
| **Fused Bias + ReLU + Residual** | Pointwise fusion pattern common in transformer inference | Triton |
| **Parallel Reduction** | Sum/max reduction — naive → warp-shuffle optimized, with progressive optimization analysis | CUDA |
| **1D Conv + Halo Exchange** | Shared memory convolution with ghost cell communication | CUDA |

Each kernel includes build instructions, benchmark tables vs baselines, Nsight Compute profiling notes, and roofline classification. See [`cuda-kernels`](https://github.com/Pupking/cuda-kernels).

---
-->
### Research

**Block-Sparse Tensor Contraction on GPUs** — MS Thesis ([NC State Repository](https://repository.lib.ncsu.edu/items/0b623de7-9602-43b2-877f-bfa7eebf1783))
CUTLASS-based pipeline for block-sparse tensor contractions in DMRG quantum simulation. 3.01× mean / 6.06× max speedup over cuTENSOR 2.5.0 block-sparse API across 20 workloads. FP64 Tensor Core config reaching ~95% of H100 peak. Code available upon paper acceptance.
*Advisor: Dr. Jiajia Li | Collaborators: NC State (Zecheng Li), Flatiron Institute (Miles Stoudenmire, Karl Pierce)*

**Publications:**
- "Image compression using quantum wavelet transforms" — *SPIE Quantum Computing, Communication, and Simulation V*, 2025
- "On-board classification of underwater images using hybrid classical-quantum CNN" — *Quantum Machine Intelligence*, 2024

---

### Key Results

| Project | Result | Hardware |
|---------|--------|----------|
| Block-sparse TC pipeline | 3.01× mean / 6.06× max over cuTENSOR | H100 |
| FP64 Tensor Core tuning | ~95% of 60 TFLOPS peak (3,000+ configs) | H100 |
| Dense TC characterization | 25.6× GETT mode-ordering sensitivity | H100 |
| Mode reordering optimization | 1.5–2.1× on 3D–5D contractions | A100/H100 |
| GPGPU-Sim cache study | 17% L1 pressure reduction | Simulated GPU |
| AUV sensor fusion | IMU/DVL/camera at 10–15 Hz | Jetson Xavier |

---

### Tech Stack

![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C++-00599C?style=flat&logo=cplusplus&logoColor=white)
![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-064F8C?style=flat&logo=cmake&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=black)
![MPI](https://img.shields.io/badge/MPI-003366?style=flat)
![OpenMP](https://img.shields.io/badge/OpenMP-5C8DBC?style=flat)
![SYCL](https://img.shields.io/badge/SYCL-003366?style=flat)

---

### Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/shr-deverapalli/)
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=flat&logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=tcn7np0AAAAJ&hl=en)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:sdevera@ncsu.edu)
