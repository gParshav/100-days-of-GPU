#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define M 256
#define N 128
#define BLOCK_SIZE 64

void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
    }
}

void elu_cpu(float* input, float* output, int m, int n, float alpha) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            output[idx] = input[idx] > 0.0f ? input[idx] : alpha * (expf(input[idx]) - 1.0f);
        }
    }
}

__global__ void elu_gpu(float* input, float* output, int m, int n, float alpha) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    for (int j = 0; j < n; j++) {
        int idx = row * n + j;
        output[idx] = input[idx] > 0.0f ? input[idx] : alpha * (expf(input[idx]) - 1.0f);
    }
}

int main() {
    float *h_A, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_C;
    float alpha = 1.0f;

    int size_A = M * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    h_A = (float*)malloc(size_A);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);

    srand(time(NULL));
    init_matrix(h_A, M, N);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, 1);
    dim3 gridDim((M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    elu_gpu<<<gridDim, blockDim>>>(d_A, d_C, M, N, alpha);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    elu_cpu(h_A, h_C_cpu, M, N, alpha);

    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-4) {
            correct = false;
            break;
        }
    }
    
    printf("Results are %s\n", correct ? "correct" : "incorrect");

    free(h_A);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_C);

    return 0;
}
