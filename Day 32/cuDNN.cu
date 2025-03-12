#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define BATCH 2       // Batch size
#define CHANNELS 3    // Number of channels
#define HEIGHT 4      // Height of input
#define WIDTH 4       // Width of input

void init_data(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // Random values between -1 and 1
    }
}

int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    int size = BATCH * CHANNELS * HEIGHT * WIDTH;
    
    // Allocate host memory
    float *h_input = (float*)malloc(size * sizeof(float));
    float *h_output = (float*)malloc(size * sizeof(float));
    init_data(h_input, size);

    // Allocate device memory
    float *d_input, *d_output, *d_mean, *d_variance, *d_gamma, *d_beta;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    cudaMalloc(&d_mean, CHANNELS * sizeof(float));
    cudaMalloc(&d_variance, CHANNELS * sizeof(float));
    cudaMalloc(&d_gamma, CHANNELS * sizeof(float));
    cudaMalloc(&d_beta, CHANNELS * sizeof(float));

    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Create tensor descriptor
    cudnnTensorDescriptor_t inputDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH, CHANNELS, HEIGHT, WIDTH);

    // Perform Batch Normalization
    float alpha = 1.0f, beta = 0.0f;
    cudnnBatchNormalizationForwardInference(
        cudnn, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
        inputDesc, d_input, inputDesc, d_output,
        inputDesc, d_gamma, d_beta, d_mean, d_variance, 1e-5);

    // Copy output back
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroy(cudnn);
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mean);
    cudaFree(d_variance);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    return 0;
}