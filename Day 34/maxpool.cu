#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256
#define N 128 

#define BLOCK_SIZE 32

__global__ void maxpool2d_gpu(float* input, float* output, int m, int n, int pool_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_m = m / pool_size;
    int out_n = n / pool_size;

    if (row < out_m && col < out_n) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < pool_size; i++) {
            for (int j = 0; j < pool_size; j++) {
                int cur_row = row * pool_size + i;
                int cur_col = col * pool_size + j;
                if (cur_row < m && cur_col < n) {
                    max_val = fmaxf(max_val, input[cur_row * n + cur_col]);
                }
            }
        }
        output[row * out_n + col] = max_val;
    }
}

void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output;

    int pool_size = 2;
    int size_input = M * N * sizeof(float);
    int size_output = (M / pool_size) * (N / pool_size) * sizeof(float);

    h_input = (float*)malloc(size_input);
    h_output = (float*)malloc(size_output);

    srand(time(NULL));
    init_matrix(h_input, M, N);

    cudaMalloc(&d_input, size_input);
    cudaMalloc(&d_output, size_output);

    cudaMemcpy(d_input, h_input, size_input, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N / pool_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (M / pool_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    maxpool2d_gpu<<<gridDim, blockDim>>>(d_input, d_output, M, N, pool_size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, size_output, cudaMemcpyDeviceToHost);

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}