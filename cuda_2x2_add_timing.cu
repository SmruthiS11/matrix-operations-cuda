#include <iostream>
#include <cuda_runtime.h>

#define N 2  // 2x2 matrix

__global__ void matrixAdd(float* A, float* B, float* C) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    int idx = row * N + col;

    C[idx] = A[idx] + B[idx];
}

void cpuAdd(float* A, float* B, float* C) {
    for (int i = 0; i < N * N; ++i)
        C[i] = A[i] + B[i];
}

int main() {
    float A[N * N] = {1, 2, 3, 4};
    float B[N * N] = {5, 6, 7, 8};
    float C_cpu[N * N], C_gpu[N * N];

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * N * N);
    cudaMalloc(&d_B, sizeof(float) * N * N);
    cudaMalloc(&d_C, sizeof(float) * N * N);

    cudaMemcpy(d_A, A, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    // CPU Timing
    cudaEvent_t cpu_start, cpu_end;
    cudaEventCreate(&cpu_start);
    cudaEventCreate(&cpu_end);
    cudaEventRecord(cpu_start);

    cpuAdd(A, B, C_cpu);

    cudaEventRecord(cpu_end);
    cudaEventSynchronize(cpu_end);
    float cpu_ms;
    cudaEventElapsedTime(&cpu_ms, cpu_start, cpu_end);

    // GPU Timing
    dim3 threadsPerBlock(N, N);
    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);
    cudaEventRecord(gpu_start);

    matrixAdd<<<1, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_end);

    cudaMemcpy(C_gpu, d_C, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

    // Print Results
    std::cout << "CPU result:\n";
    for (int i = 0; i < N * N; ++i) std::cout << C_cpu[i] << (i % N == N - 1 ? "\n" : " ");
    std::cout << "\nGPU result:\n";
    for (int i = 0; i < N * N; ++i) std::cout << C_gpu[i] << (i % N == N - 1 ? "\n" : " ");

    std::cout << "\nCPU time: " << cpu_ms << " ms\n";
    std::cout << "GPU time: " << gpu_ms << " ms\n";
    std::cout << "Speedup: " << cpu_ms / gpu_ms << "x\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(cpu_start);
    cudaEventDestroy(cpu_end);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_end);

    return 0;
}
