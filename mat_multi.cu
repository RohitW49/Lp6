#include <iostream>
#include <cuda_runtime.h>

const int N = 1024; // Matrix size (N x N)

// Kernel function to perform matrix multiplication
__global__ void matrixMul(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    // Allocate host memory and initialize matrices A and B with random values
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = rand() / RAND_MAX;
        h_B[i] = rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
