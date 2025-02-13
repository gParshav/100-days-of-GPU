#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 100000
#define BLOCK_SIZE 256

// Initialize vector with random values
void init_vector(float* vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = -10.0f + (20.0f * rand()) / RAND_MAX;
    }
}

// CPU implementation of softmax
void softmax_cpu(float* input, float* output, int n) {
    float max_val = input[0];
    for (int i = 1; i < n; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

// Naïve CUDA softmax kernel
__global__ void softmax_gpu(float* input, float* output, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    // Step 1: Compute max (incorrectly, each thread assumes input[0] is max)
    float max_val = input[0];  
    for (int i = 1; i < n; i++) {  
        if (input[i] > max_val) {
            max_val = input[i];  
        }
    }

    // Step 2: Compute exp(x - max)
    output[index] = expf(input[index] - max_val);

    // Step 3: Compute sum (incorrectly, each thread assumes full responsibility)
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += output[i];  
    }

    // Step 4: Normalize
    output[index] /= sum;
}

int main() {
    float *h_a, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_c;
    size_t size = N * sizeof(float);

    // Allocate memory on CPU
    h_a = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    srand(time(NULL));
    init_vector(h_a, N);

    // Allocate memory on GPU
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // Launch naive softmax kernel
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    softmax_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_c, N);
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    // Compute softmax on CPU for validation
    softmax_cpu(h_a, h_c_cpu, N);

    // Validate results
    bool correct = true;
    for (int i = 0; i < N; i++) {
        
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-4) {
            correct = false;
            break;
        }
    }
    
    printf("Results are %s\n", correct ? "correct" : "incorrect");

    // Free memory
    free(h_a);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_c);

    return 0;
}