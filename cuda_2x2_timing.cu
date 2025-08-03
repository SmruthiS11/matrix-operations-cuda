#include <iostream> //including standard input-output functions
#include <cuda_runtime.h> // including CUDA runtime

#define N 2  // 2x2 matrix

__global__ void matrixMultiply(float* A, float* B, float* C) { //declaring the CUDA kernel that runs on GPU
    int row = threadIdx.y; //thread indices for row and column
    int col = threadIdx.x;

    float value = 0; //initialising C as 0
    for (int k = 0; k < N; ++k) { //defining the iteration over row and column for finding the dot product
        value += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = value; // final value inC
}

void cpuMultiply(float* A, float* B, float* C) { //multiplying A and B on CPU and storing it in C
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float val = 0;
            for (int k = 0; k < N; ++k)
                val += A[i * N + k] * B[k * N + j];
            C[i * N + j] = val;
        }
}

int main() { //defining the matrices
    float A[N * N] = {1, 2, 3, 4};   // Row-major: [ [1, 2], [3, 4] ]
    float B[N * N] = {5, 6, 7, 8};   // Row-major: [ [5, 6], [7, 8] ]
    float C_cpu[N * N], C_gpu[N * N];

    float *d_A, *d_B, *d_C; //device pointers

    cudaMalloc(&d_A, sizeof(float) * N * N); //memory allocation on GPU
    cudaMalloc(&d_B, sizeof(float) * N * N);
    cudaMalloc(&d_C, sizeof(float) * N * N);

    cudaMemcpy(d_A, A, sizeof(float) * N * N, cudaMemcpyHostToDevice); //copying from CPU to GPU memory
    cudaMemcpy(d_B, B, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    // CPU Timing
    cudaEvent_t cpu_start, cpu_end;
    cudaEventCreate(&cpu_start);
    cudaEventCreate(&cpu_end);
    cudaEventRecord(cpu_start);

    cpuMultiply(A, B, C_cpu);

    cudaEventRecord(cpu_end);
    cudaEventSynchronize(cpu_end);
    float cpu_ms;
    cudaEventElapsedTime(&cpu_ms, cpu_start, cpu_end);

    // GPU Timing
    dim3 threadsPerBlock(N, N); //thread launched
    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);
    cudaEventRecord(gpu_start);

    matrixMultiply<<<1, threadsPerBlock>>>(d_A, d_B, d_C); //launches 4 threads and each calculate one element of C

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
