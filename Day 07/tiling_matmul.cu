#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256 
#define K 512
#define N 128
#define BLOCK_SIZE 32

// A is M*K
// B is K*N

// A*B will be M*N



__global__ void tiled_matmul(float* A, float* B, float* C, int m, int k, int n) {
    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        if (row < M && tile * BLOCK_SIZE + tx < K)
            sharedA[ty][tx] = A[row * K + tile * BLOCK_SIZE + tx];
        else
            sharedA[ty][tx] = 0.0f;
        
        if (col < N && tile * BLOCK_SIZE + ty < K)
            sharedB[ty][tx] = B[(tile * BLOCK_SIZE + ty) * N + col];
        else
            sharedB[ty][tx] = 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum += sharedA[ty][k] * sharedB[k][tx];
        
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}

void matmul_cpu(float* A, float* B, float*C, int m, int k, int n){
    
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            float sum = 0.0f;
            for(int l=0;l<k;l++) {
                sum+= A[i*k+l]*B[l*n+j];
            }  
            C[i*n+j] = sum;
        }
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

    float *h_A, *h_B, *h_C_cpu, *h_C_gpu; //These will be stored on the CPU
    float *d_A, *d_B, *d_C; //These will be stored on the GPU

    int size_A = M*K*sizeof(float);
    int size_B = K*N*sizeof(float);
    int size_C = M*N*sizeof(float);

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);

    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    //Lets see for X first
    // In the naive kernel, I have rows handled by threads in the X direction. We have M rows in all and therefore m threads in the x direction
    // We have to fit total M threads along the X-axis. Threads in a block in X = BLOCK_SIZE. Number of blocks in X = (M+BLOCK_SIZE-1)/BLOCK_SIZE
    // We have to fit total N threads along the Y-axis. Threads in a block in Y = BLOCK_SIZE. Number of blocks in Y = (N+BLOCK_SIZE-1)/BLOCK_SIZE

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); //These are number of threads in x and y inside the block
    dim3 gridDim((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);

    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
    tiled_matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;





}

