#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

#define N 1024  // Matrix size: N x N

__global__ void matMulKernel(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

void cpuMatMul(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0;
            for (int k = 0; k < n; ++k)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
}

int main() {
    size_t bytes = N * N * sizeof(float);

    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C_cpu = new float[N * N];
    float* C_gpu = new float[N * N];

    srand(time(0));
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // CPU timing
    cudaEvent_t cpu_start, cpu_end;
    cudaEventCreate(&cpu_start);
    cudaEventCreate(&cpu_end);
    cudaEventRecord(cpu_start);

    cpuMatMul(A, B, C_cpu, N);

    cudaEventRecord(cpu_end);
    cudaEventSynchronize(cpu_end);
    float cpu_ms = 0;
    cudaEventElapsedTime(&cpu_ms, cpu_start, cpu_end);

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // GPU timing
    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);
    cudaEventRecord(gpu_start);

    matMulKernel<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);
    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_end);

    cudaMemcpy(C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

    // Compare results (optional)
    bool correct = true;
    for (int i = 0; i < N * N; ++i) {
        if (fabs(C_cpu[i] - C_gpu[i]) > 1e-3f) {
            correct = false;
            break;
        }
    }

    std::cout << "\nCPU time: " << cpu_ms << " ms\n";
    std::cout << "GPU time: " << gpu_ms << " ms\n";
    std::cout << "Speedup: " << cpu_ms / gpu_ms << "x\n";
    std::cout << "Results match: " << (correct ? "Yes" : "No") << "\n";

    // Cleanup
    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_gpu;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(cpu_start);
    cudaEventDestroy(cpu_end);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_end);

    return 0;
}
