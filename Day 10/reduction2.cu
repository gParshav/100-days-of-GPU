#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 200000
#define BLOCK_SIZE 256

void init_vector(float* vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

void reduction_cpu(float* a, float* c, int n) {
    c[0] = 0;
    for (int i = 0; i < n; i++) {
        c[0] += a[i];
    }
}

__global__ void reduction_gpu(float* a, float* c, int n) {
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float s_data[];

    
    if (index < n) {
        s_data[tid] = a[index];
        __syncthreads();

        for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
            if (tid % (2 * stride) == 0 && index + stride < n) {
                s_data[tid] += s_data[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            c[blockIdx.x] = s_data[0]; 
        }
    }
}

// CPU time measurement function
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_in, *h_out_cpu, *h_out_gpu;
    float *d_in, *d_out;

    size_t size = N * sizeof(float);

    // Allocate memory for arrays on the CPU
    h_in = (float*)malloc(size);
    h_out_cpu = (float*)malloc(size);
    h_out_gpu = (float*)malloc(size);

    srand(time(NULL));

    init_vector(h_in, N);

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Measure GPU execution time using CUDA events
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, 0);
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduction_gpu<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaDeviceSynchronize(); // Ensure the kernel finishes before measuring stop time
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start_event, stop_event);

    // Copy the result from GPU to host
    cudaMemcpy(h_out_gpu, d_out, size, cudaMemcpyDeviceToHost);

    float gpu_sum = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        gpu_sum += h_out_gpu[i];
    }

    // Calculate GFLOPS
    double gpu_gflops = 2.0 * N / (gpu_time / 1000.0) / 1e9; // gpu_time is in milliseconds, so divide by 1000

    // Calculate memory bandwidth
    double gpu_memory_bandwidth = 2.0 * N * sizeof(float) / (gpu_time / 1000.0) / 1e9; // GB/s

    // Measure CPU execution time using get_time()
    double start_cpu = get_time();
    reduction_cpu(h_in, h_out_cpu, N);
    double end_cpu = get_time();

    double cpu_time = end_cpu - start_cpu;

    printf("%f %f\n", gpu_sum, h_out_cpu[0]);

    // Print results
    printf("GPU execution time: %f seconds\n", gpu_time / 1000.0); // Convert to seconds
    printf("GPU GFLOPS: %f GFLOPS\n", gpu_gflops);
    printf("GPU memory bandwidth: %f GB/s\n", gpu_memory_bandwidth);
    printf("CPU execution time: %f seconds\n", cpu_time);

    // Clean up
    free(h_in);
    free(h_out_cpu);
    free(h_out_gpu);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return 0;
}