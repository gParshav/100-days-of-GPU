#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define N 10000  

void init_vector(float* vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; 
    }
}

void softmax_cpu(float* input, float* output, int length) {
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        output[i] = expf(input[i]);
        sum += output[i];
    }
    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

int main() {
    float *h_in, *h_out_cpu, *h_out_gpu;
    float *d_in, *d_out;

    size_t size = N * sizeof(float);
    
    h_in = (float*)malloc(size);
    h_out_cpu = (float*)malloc(size);
    h_out_gpu = (float*)malloc(size);

    srand(time(NULL));
    init_vector(h_in, N);

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // cuDNN setup
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t tensorDesc;
    cudnnCreateTensorDescriptor(&tensorDesc);
    cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, N);

    float alpha = 1.0f, beta = 0.0f;
    cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, 
                        &alpha, tensorDesc, d_in, &beta, tensorDesc, d_out);

    cudaMemcpy(h_out_gpu, d_out, size, cudaMemcpyDeviceToHost);

    // Compute softmax on CPU for validation
    softmax_cpu(h_in, h_out_cpu, N);

    // Validate the results
    bool match = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_out_cpu[i] - h_out_gpu[i]) > 1e-5) { 
            match = false;
            break;
        }
    }

    if (match) {
        printf("CPU and GPU softmax outputs match.\n");
    } else {
        printf("CPU and GPU softmax outputs do not match.\n");
    }

    // Cleanup
    cudnnDestroyTensorDescriptor(tensorDesc);
    cudnnDestroy(cudnn);

    free(h_in);
    free(h_out_cpu);
    free(h_out_gpu);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}