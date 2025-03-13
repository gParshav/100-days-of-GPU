#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define N 10000
#define DROPOUT_PROB 0.5f  

void init_vector(float* vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX; 
    }
}

int main() {
    float *h_in, *h_out_gpu;
    float *d_in, *d_out, *d_states;
    size_t size = N * sizeof(float);

    h_in = (float*)malloc(size);
    h_out_gpu = (float*)malloc(size);
    srand(time(NULL));
    init_vector(h_in, N);

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnDropoutDescriptor_t dropoutDesc;
    cudnnCreateDropoutDescriptor(&dropoutDesc);

    size_t stateSize;
    cudnnDropoutGetStatesSize(cudnn, &stateSize);
    cudaMalloc(&d_states, stateSize);

    cudnnSetDropoutDescriptor(dropoutDesc, cudnn, DROPOUT_PROB, d_states, stateSize, 0);

    cudnnTensorDescriptor_t tensorDesc;
    cudnnCreateTensorDescriptor(&tensorDesc);
    cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, N);

    cudnnDropoutForward(cudnn, dropoutDesc, tensorDesc, d_in, tensorDesc, d_out, NULL, 0);

    cudaMemcpy(h_out_gpu, d_out, size, cudaMemcpyDeviceToHost);

    cudnnDestroyDropoutDescriptor(dropoutDesc);
    cudnnDestroyTensorDescriptor(tensorDesc);
    cudnnDestroy(cudnn);

    free(h_in);
    free(h_out_gpu);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_states);

    printf("Dropout applied successfully!\n");

    return 0;
}