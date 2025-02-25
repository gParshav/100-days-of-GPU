#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define M 256
#define N 128
#define BLOCK_SIZE 64

// Initialize vector with random values
void init_vector(float* vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = -10.0f + (20.0f * rand()) / RAND_MAX;
    }
}

void init_matrix(float* mat, int rows, int cols){
    for(int i=0;i<rows*cols;i++){
        mat[i] = (float)rand() / RAND_MAX;
    }
}


// CPU implementation of softmax
void softmax_cpu(float* input, float* output, int m, int n) {
    
    for(int i=0;i<m;i++){
        float sum = 0.0f;
        float max_val = 0.0f;
        for (int j = 0; j < n; j++) {
            max_val = max(max_val, input[i*n+j]);
        }

        for (int j = 0; j < n; j++) {
            sum += expf(input[i*n+j] - max_val);
        }

        for (int j = 0; j < n; j++) {
            output[i*n+j] = expf(input[i*n+j] - max_val) / sum;
        }
    }
    
}

// Naïve CUDA softmax kernel
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

__global__ void online_softmax_gpu(float* input, float* output, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    float sum = 0.0f;
    float max_val = 0.0f;
    for (int j = 0; j < n; j++) {
        float curr = input[row*n+j];
        if(curr>max_val){
            sum = sum*expf(max_val - curr);
            max_val = curr;
        }

        sum+=expf(curr - max_val);
    }

    for (int j = 0; j < n; j++) {
        output[row*n+j] = expf(input[row*n+j] - max_val) / sum;
    }
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

    dim3 blockDim(BLOCK_SIZE, 1); //These are number of threads in x and y inside the block
    dim3 gridDim((M+BLOCK_SIZE-1)/BLOCK_SIZE, 1);


    online_softmax_gpu<<<gridDim, blockDim>>>(d_A, d_C, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    // Compute softmax on CPU for validation
    softmax_cpu(h_A, h_C_cpu, M, N);

    // Validate results
    bool correct = true;
    for (int i = 0; i < N; i++) {
        
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-4) {
            correct = false;
            break;
        }
    }
    
    printf("Results are %s\n", correct ? "correct" : "incorrect");

    // Free memory
    free(h_A);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_C);

    return 0;

}