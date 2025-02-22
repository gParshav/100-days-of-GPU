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

__global__ void conv2d_gpu(float* image, float* filter, float* C, int m, int n, int k){
    
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int h = (k-1)/2;
    int sr = row-h;
    int sc = col-h;
    
    if(row<m && col<n){
        float sum = 0.0f;
        for(int i=0;i<k;i++){
            for(int j=0;j<k;j++){
                int cur_sr = sr + i;  // Reset for each iteration
                int cur_sc = sc + j;
                if(cur_sr >= 0 && cur_sc >= 0 && cur_sr < m && cur_sc < n) {
                    sum += image[cur_sr * n + cur_sc] * filter[i * k + j];
                }
            }
        }
        // printf("%f\n", sum);
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

    float *h_image, *h_filter, *h_C_gpu; //These will be stored on the CPU
    float *d_image, *d_filter, *d_C; //These will be stored on the GPU

    int size_image = M*N*sizeof(float);
    int size_filter = K*K*sizeof(float);

    h_image = (float*)malloc(size_image);
    h_filter = (float*)malloc(size_filter);
    h_C_gpu = (float*)malloc(size_image);

    srand(time(NULL));
    init_matrix(h_image, M, N);
    init_matrix(h_filter, K, K);

    cudaMalloc(&d_image, size_image);
    cudaMalloc(&d_filter, size_filter);
    cudaMalloc(&d_C, size_image);

    cudaMemcpy(d_image, h_image, size_image, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, size_filter, cudaMemcpyHostToDevice);

    //Lets see for X first
    // In the naive kernel, I have rows handled by threads in the X direction. We have M rows in all and therefore m threads in the x direction
    // We have to fit total M threads along the X-axis. Threads in a block in X = BLOCK_SIZE. Number of blocks in X = (M+BLOCK_SIZE-1)/BLOCK_SIZE
    // We have to fit total N threads along the Y-axis. Threads in a block in Y = BLOCK_SIZE. Number of blocks in Y = (N+BLOCK_SIZE-1)/BLOCK_SIZE

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); //These are number of threads in x and y inside the block
    dim3 gridDim((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);

    conv2d_gpu<<<gridDim, blockDim>>>(d_image, d_filter, d_C, M, N, K);
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

