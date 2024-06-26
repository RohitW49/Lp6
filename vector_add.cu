#include <iostream>
#include <cuda_runtime.h>

__global__ void addVectors(int *A, int *B, int *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int n = 1000000;
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    // Host arrays
    int *A, *B, *C;
    cudaMallocHost(&A, n * sizeof(int));
    cudaMallocHost(&B, n * sizeof(int));
    cudaMallocHost(&C, n * sizeof(int));

    // Initialize vectors A and B
    for (int i = 0; i < n; ++i) {
        A[i] = i;
        B[i] = i * 2;
    }

    // Device arrays
    int *dev_A, *dev_B, *dev_C;
    cudaMalloc(&dev_A, n * sizeof(int));
    cudaMalloc(&dev_B, n * sizeof(int));
    cudaMalloc(&dev_C, n * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(dev_A, A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    addVectors<<<numBlocks, blockSize>>>(dev_A, dev_B, dev_C, n);
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(C, dev_C, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < 10; ++i) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}
