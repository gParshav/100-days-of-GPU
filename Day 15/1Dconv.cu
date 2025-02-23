#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256
#define K 3

#define BLOCK_SIZE 32


__global__ void conv1d_gpu(float* image, float* filter, float* C, int m, int k){
    
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int h = (k-1)/2;
    int si = idx-h;
    
    if(idx<m){
        float sum = 0.0f;
        for(int i=0;i<k;i++){
            int cur_si = si + i;
            if(cur_si >= 0 && cur_si < m) {
                sum += image[cur_si] * filter[i];
            }
        }

        C[idx] = sum;
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

    float *h_image, *h_filter, *h_C_gpu; //These will be stored on the CPU
    float *d_image, *d_filter, *d_C; //These will be stored on the GPU

    int size_image = M*sizeof(float);
    int size_filter = K*sizeof(float);

    h_image = (float*)malloc(size_image);
    h_filter = (float*)malloc(size_filter);
    h_C_gpu = (float*)malloc(size_image);

    srand(time(NULL));
    init_matrix(h_image, M, 1);
    init_matrix(h_filter, K, 1);

    cudaMalloc(&d_image, size_image);
    cudaMalloc(&d_filter, size_filter);
    cudaMalloc(&d_C, size_image);

    cudaMemcpy(d_image, h_image, size_image, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, size_filter, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, 1); //These are number of threads in x and y inside the block
    dim3 gridDim((M+BLOCK_SIZE-1)/BLOCK_SIZE, 1);

    conv1d_gpu<<<gridDim, blockDim>>>(d_image, d_filter, d_C, M, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_gpu, d_C, size_image, cudaMemcpyDeviceToHost);
    free(h_image);
    free(h_filter);
    free(h_C_gpu);
    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_C);

    return 0;





}

