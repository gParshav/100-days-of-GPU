#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256
#define N 128
#define BLOCK_SIZE 32

// A is M*N

void transpose_cpu(float* A, float* C, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[j * m + i] = A[i * n + j]; // Transpose operation
        }
    }
}

__global__ void transpose_gpu(float* A, float* C, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        C[col * m + row] = A[row * n + col]; // Transpose operation
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

    float *h_A, *h_C_cpu, *h_C_gpu; //These will be stored on the CPU
    float *d_A, *d_C; //These will be stored on the GPU

    int size_A = M*N*sizeof(float);
    int size_C = M*N*sizeof(float);

    h_A = (float*)malloc(size_A);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);

    srand(time(NULL));
    init_matrix(h_A, M, N);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);

    //Lets see for X first
    // In the naive kernel, I have rows handled by threads in the X direction. We have M rows in all and therefore m threads in the x direction
    // We have to fit total M threads along the X-axis. Threads in a block in X = BLOCK_SIZE. Number of blocks in X = (M+BLOCK_SIZE-1)/BLOCK_SIZE
    // We have to fit total N threads along the Y-axis. Threads in a block in Y = BLOCK_SIZE. Number of blocks in Y = (N+BLOCK_SIZE-1)/BLOCK_SIZE

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); //These are number of threads in x and y inside the block
    dim3 gridDim((M+BLOCK_SIZE-1)/BLOCK_SIZE, (N+BLOCK_SIZE-1)/BLOCK_SIZE);

    transpose_cpu(h_A, h_C_cpu, M, N);
    transpose_gpu<<<gridDim, blockDim>>>(d_A, d_C, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < M*N; i++) {
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

