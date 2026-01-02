# CUDA Matrix Multiplication and Addition

This repository contains simple CUDA programs demonstrating matrix multiplication and matrix addition, along with kernel execution timing using NVIDIA CUDA.  
Each program deals with a specific matrix size, as indicated in the filename.

The main intention behind this project was to measure GPU execution time for small and large matrices
## Files Overview

### 1. `cuda_1024x1024_timing.cu`
- Operation: Matrix Multiplication
- Matrix Size: 1024 × 1024
### 2. `cuda_2x2_timing.cu`
- Operation: Matrix Multiplication
- Matrix Size: 2 × 2
### 3. `cuda_2x2_add_timing.cu`
- **Operation:** Matrix Addition
- **Matrix Size:** 2 × 2

## Build and Run Instructions

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (`nvcc` available)
- It can be run on Google Colab with GPU
