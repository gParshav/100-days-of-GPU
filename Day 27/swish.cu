#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 10000000
#define BLOCK_SIZE 256

void init_vector(float* vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

void swish_cpu(float* x, float* y, int n) {
    for (int i = 0; i < n; i++) {
        float cdf = 0.5f * (1.0f + erf(x[i] / sqrtf(2.0f)));
        y[i] = x[i] * cdf;
    }
}

__global__ void swish_gpu(float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float cdf = 0.5f * (1.0f + erff(x[idx] / sqrtf(2.0f)));
        y[idx] = x[idx] * cdf;
    }
}

int main() {
    float *h_a, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_c;

    size_t size = N * sizeof(float);

    // Allocate memory for arrays on the CPU
    h_a = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    srand(time(NULL));
    init_vector(h_a, N);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    swish_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_c, N);
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    // Validate results
    swish_cpu(h_a, h_c_cpu, N);

    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    printf("Results are %s\n", correct ? "correct" : "incorrect");

    // Free resources
    free(h_a);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_c);

    return 0;
}
