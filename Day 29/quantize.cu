#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>
#include <float.h>

#define N 200000
#define BLOCK_SIZE 256

// Function to initialize the input vector with random values
void init_vector(float* vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// CPU implementation of the quantization
void quantize_cpu(float *input, size_t length, float S, uint8_t Z, uint8_t *output) {
    for (int i = 0; i < length; i++) {
        output[i] = (uint8_t)roundf(input[i] / S + Z); 
    }
}

// GPU kernel for reduction of min and max values
__global__ void reduction_min_max(float *input, size_t length, float *d_min, float *d_max) {

    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float smax[];
    extern __shared__ float smin[];

    smin[tid] = (idx < length) ? input[idx] : FLT_MAX;
    smax[tid] = (idx < length) ? input[idx] : -FLT_MAX;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smax[tid] = max(smax[tid], smax[tid + stride]);
            smin[tid] = min(smin[tid], smin[tid + stride]);
        }

        __syncthreads();
    }

    if (tid == 0) {
        d_min[blockIdx.x] = smin[0];
        d_max[blockIdx.x] = smax[0];
    }
    
}

// GPU kernel for quantization
__global__ void quantize_gpu(float *input, size_t length, float S, uint8_t Z, uint8_t *output) {
    int stride = gridDim.x * blockDim.x;
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < length; idx += stride) {
        output[idx] = __float2int_rn(input[idx] / S + Z); // Converts float to nearest int
    }
}

// CPU time measurement function
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_in;
    uint8_t *h_out_cpu, *h_out_gpu;
    float *d_in, *d_min, *d_max;
    uint8_t *d_out;

    size_t size = N * sizeof(float);
    
    h_in = (float*)malloc(size);
    h_out_cpu = (uint8_t*)malloc(N * sizeof(uint8_t));
    h_out_gpu = (uint8_t*)malloc(N * sizeof(uint8_t));

    srand(time(NULL));

    init_vector(h_in, N);
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // dmin and dmax contains min and maxes of every block
    //global min and max can be calculated by iterating them
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_min, num_blocks * sizeof(float)); 
    cudaMalloc(&d_max, num_blocks * sizeof(float)); 
    cudaMalloc(&d_out, N * sizeof(uint8_t)); 

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    reduction_min_max<<<num_blocks, BLOCK_SIZE>>>(d_in, N, d_min, d_max);
    cudaDeviceSynchronize(); 

    float *h_min = (float*)malloc(num_blocks * sizeof(float));
    float *h_max = (float*)malloc(num_blocks * sizeof(float));

    cudaMemcpy(h_min, d_min, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max, d_max, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;

    for (int i = 0; i < num_blocks; i++) {
        min_val = fminf(min_val, h_min[i]);
        max_val = fmaxf(max_val, h_max[i]);
    }

    float S = (max_val - min_val) / 255.0f;
    uint8_t Z = roundf(min_val / S);

    quantize_gpu<<<num_blocks, BLOCK_SIZE>>>(d_in, N, S, Z, d_out);
    cudaDeviceSynchronize(); 
    cudaMemcpy(h_out_gpu, d_out, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    quantize_cpu(h_in, N, S, Z, h_out_cpu); // Using same scale S and zero point Z

    bool match = true;
    for (int i = 0; i < N; i++) {
        if (h_out_cpu[i] != h_out_gpu[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        printf("CPU and GPU outputs match.\n");
    } else {
        printf("CPU and GPU outputs do not match.\n");
    }

    // Clean up
    free(h_in);
    free(h_out_cpu);
    free(h_out_gpu);
    free(h_min);
    free(h_max);

    cudaFree(d_in);
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_out);

    return 0;
}