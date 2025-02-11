#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 512 
#define K 512
#define N 512
#define BLOCK_SIZE 32

// A is M*K
// B is K*N

// A*B will be M*N

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

__global__ void matmul_gpu(float* A, float* B, float* C, int m, int k, int n){
    
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

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
    // We have to fit total M threads along the Y-axis. Threads in a block in Y = BLOCK_SIZE. Number of blocks in Y = (M+BLOCK_SIZE-1)/BLOCK_SIZE
    // We have to fit total N threads along the X-axis. Threads in a block in X = BLOCK_SIZE. Number of blocks in X = (N+BLOCK_SIZE-1)/BLOCK_SIZE

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); //These are number of threads in x and y inside the block
    dim3 gridDim((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);

    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
    matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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

