#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256
#define N 128
#define BLOCK_SIZE 32


// A*B will be M*N

__global__ void layernorm_gpu(float* A, float* C, int m, int n){
    
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if(row>=m){
        return;
    }
    
    float sum = 0.0f;
    for(int l=0;l<n;l++){
        int idx = row*n+l;
        sum+=A[idx];
    }

    float mean = sum/n;
    float diff_sum = 0.0f;
    for(int l=0;l<n;l++){
        int idx = row*n+l;
        diff_sum+=(A[idx]-mean)*(A[idx]-mean);
    }

    float var = diff_sum/n;
    float stddev = sqrt(var);  

    for(int l=0;l<n;l++){
        int idx = row*n+l;
        C[idx] = (A[idx] - mean) / stddev;     
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

    float *h_A, *h_C_gpu; //These will be stored on the CPU
    float *d_A, *d_C; //These will be stored on the GPU

    int size_A = M*N*sizeof(float);
    int size_C = M*N*sizeof(float);

    h_A = (float*)malloc(size_A);
    h_C_gpu = (float*)malloc(size_C);

    srand(time(NULL));
    init_matrix(h_A, M, N);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, 1); //These are number of threads in x and y inside the block
    dim3 gridDim((M+BLOCK_SIZE-1)/BLOCK_SIZE, 1);

    layernorm_gpu<<<gridDim, blockDim>>>(d_A, d_C, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    free(h_A);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_C);

    return 0;





}

