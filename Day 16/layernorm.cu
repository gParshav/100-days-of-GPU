#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define M 25600
#define N 12800
#define BLOCK_SIZE 32

void layernorm_cpu(float* A, float* C, int m, int n) {
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j];
        }
        float mean = sum / n;
        float diff_sum = 0.0f;
        for (int j = 0; j < n; j++) {
            diff_sum += (A[i * n + j] - mean) * (A[i * n + j] - mean);
        }
        float var = diff_sum / n;
        float stddev = sqrt(var);
        for (int j = 0; j < n; j++) {
            C[i * n + j] = (A[i * n + j] - mean) / stddev;
        }
    }
}

__global__ void layernorm_gpu(float* A, float* C, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    
    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        sum += A[row * n + j];
    }
    float mean = sum / n;
    float diff_sum = 0.0f;
    for (int j = 0; j < n; j++) {
        diff_sum += (A[row * n + j] - mean) * (A[row * n + j] - mean);
    }
    float var = diff_sum / n;
    float stddev = sqrt(var);
    for (int j = 0; j < n; j++) {
        C[row * n + j] = (A[row * n + j] - mean) / stddev;
    }
}

void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_A, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_C;

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

    double start_cpu = get_time();
    layernorm_cpu(h_A, h_C_cpu, M, N);
    double end_cpu = get_time();
    printf("CPU Time: %f seconds\n", end_cpu - start_cpu);

    double start_gpu = get_time();
    layernorm_gpu<<<gridDim, blockDim>>>(d_A, d_C, M, N);
    cudaDeviceSynchronize();
    double end_gpu = get_time();
    printf("GPU Time: %f seconds\n", end_gpu - start_gpu);

    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < M * N; i++) {
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
