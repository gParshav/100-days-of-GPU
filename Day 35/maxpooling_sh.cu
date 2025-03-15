#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256
#define N 128 
#define BLOCK_SIZE 32
#define POOL_SIZE 2

__global__ void maxpool2d_gpu(float* input, float* output, int m, int n, int pool_size) {
    __shared__ float tile[BLOCK_SIZE + POOL_SIZE - 1][BLOCK_SIZE + POOL_SIZE - 1];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_m = m / pool_size;
    int out_n = n / pool_size;
    
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;
    
    if (row < m && col < n) {
        tile[local_row][local_col] = input[row * n + col];
    }
    __syncthreads();
    
    if (local_row % pool_size == 0 && local_col % pool_size == 0 && row / pool_size < out_m && col / pool_size < out_n) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < pool_size; i++) {
            for (int j = 0; j < pool_size; j++) {
                max_val = fmaxf(max_val, tile[local_row + i][local_col + j]);
            }
        }
        output[(row / pool_size) * out_n + (col / pool_size)] = max_val;
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

    int size_input = M * N * sizeof(float);
    int size_output = (M / POOL_SIZE) * (N / POOL_SIZE) * sizeof(float);

    h_input = (float*)malloc(size_input);
    h_output = (float*)malloc(size_output);

    srand(time(NULL));
    init_matrix(h_input, M, N);

    cudaMalloc(&d_input, size_input);
    cudaMalloc(&d_output, size_output);

    cudaMemcpy(d_input, h_input, size_input, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N / POOL_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (M / POOL_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    maxpool2d_gpu<<<gridDim, blockDim>>>(d_input, d_output, M, N, POOL_SIZE);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, size_output, cudaMemcpyDeviceToHost);

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}