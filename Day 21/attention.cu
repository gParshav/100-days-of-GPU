#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256 
#define K 512
#define N 128
#define BLOCK_SIZE 32


// A*B
__global__ void matmul_gpu(float* A, float* B, float* C, int m, int k, int n){
    
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    //You are in thread(row, col) that finds element in C at (row, col)
    //Need to make sure that row and col are within bounds in C
    // Dimensions of C is m*n
    if(row<m && col<n){
        float sum = 0.0f;
        for(int l=0;l<k;l++){
            sum+=A[row*k+l] * B[l*n+col];
        }
        
        C[row*n+col] = sum;
    }
}

// A*BT
__global__ void matmul_gpu2(float* A, float* B, float* C, int m, int k, int n){
    
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    //You are in thread(row, col) that finds element in C at (row, col)
    //Need to make sure that row and col are within bounds in C
    // Dimensions of C is m*n
    if(row<m && col<n){
        float sum = 0.0f;
        for(int l=0;l<k;l++){
            sum+=A[row*k+l] * B[col*n+l];
        }
        
        C[row*n+col] = sum;
    }
}

__global__ void softmax_gpu(float* input, float* output, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    float sum = 0.0f;
    float max_val = 0.0f;
    for (int j = 0; j < n; j++) {
        max_val = max(max_val, input[row*n+j]);
    }

    for (int j = 0; j < n; j++) {
        sum += expf(input[row*n+j] - max_val);
    }

    for (int j = 0; j < n; j++) {
        output[row*n+j] = expf(input[row*n+j] - max_val) / sum;
    }
}


__global__ void scale(float* matrix, int size, float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        matrix[idx] *= scale_factor;
    }
}

void init_matrix(float* mat, int rows, int cols){
    for(int i=0;i<rows*cols;i++){
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_A, *h_B, *h_C_gpu, *h_SoftmaxOutput;
    float *d_A, *d_B, *d_C, *d_SoftmaxInput, *d_SoftmaxOutput, *d_Result;

    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);
    int size_Softmax = M * N * sizeof(float);

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_gpu = (float*)malloc(size_C);
    h_SoftmaxOutput = (float*)malloc(size_Softmax);

    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMalloc(&d_SoftmaxInput, size_C);
    cudaMalloc(&d_SoftmaxOutput, size_Softmax);
    cudaMalloc(&d_Result, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    softmax_gpu<<<(M + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_C, d_SoftmaxOutput, M, N);
    cudaDeviceSynchronize();

    matmul_gpu2<<<gridDim, blockDim>>>(d_SoftmaxOutput, d_B, d_Result, M, N, N);
    cudaDeviceSynchronize();

    float scale_factor = 1.0f;
    scale<<<(M * N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_Result, M * N, scale_factor);
    cudaDeviceSynchronize();

    cudaMemcpy(h_SoftmaxOutput, d_Result, size_Softmax, cudaMemcpyDeviceToHost);

    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_SoftmaxOutput);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_SoftmaxInput);
    cudaFree(d_SoftmaxOutput);
    cudaFree(d_Result);

    return 0;
}
