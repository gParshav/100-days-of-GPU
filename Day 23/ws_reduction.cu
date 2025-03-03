#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000000
#define WARP_SIZE 32
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

__inline__ __device__ float warpReduceSum(float val){

    for(int offset = WARP_SIZE/2;offset>0;offset/=2) {
        val+=__shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    return val;

}

__global__ void reduction_gpu(float* input, float* output, int n) {
    
    __shared__ float sm[BLOCK_SIZE/WARP_SIZE];

    int tid = threadIdx.x+blockDim.x*blockIdx.x;
    // Lane is the position of each thread inside the warp
    int lane = threadIdx.x%WARP_SIZE;
    int warpId = threadIdx.x/WARP_SIZE;

    float val = (tid < n) ? input[tid] : 0.0f;
    // the val variable of thread 0 in every will have the sum of that warp
    val = warpReduceSum(val);
    if(lane==0){
        sm[warpId] = val;
    }

    __syncthreads();

    if (warpId == 0) {
        val = (lane < (BLOCK_SIZE / WARP_SIZE)) ? sm[lane] : 0.0f;
        val = warpReduceSum(val);
    }

    if (threadIdx.x == 0) output[blockIdx.x] = val;


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
    h_out_cpu = (float*)malloc(sizeof(float));
    
    // Calculate number of blocks needed
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    h_out_gpu = (float*)malloc(num_blocks * sizeof(float));
    
    srand(time(NULL));
    
    init_vector(h_in, N);
    
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, num_blocks * sizeof(float));
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    // Measure GPU execution time using CUDA events
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    cudaEventRecord(start_event, 0);
    reduction_gpu<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaDeviceSynchronize(); // Ensure the kernel finishes before measuring stop time
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start_event, stop_event);
    
    // Copy the result from GPU to host
    cudaMemcpy(h_out_gpu, d_out, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float gpu_sum = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        gpu_sum += h_out_gpu[i];
    }
    
    // Calculate GFLOPS
    double gpu_gflops = (double)N / (gpu_time / 1000.0) / 1e9; // gpu_time is in milliseconds
    
    // Calculate memory bandwidth
    double gpu_memory_bandwidth = 2.0 * N * sizeof(float) / (gpu_time / 1000.0) / 1e9; // GB/s
    
    // Measure CPU execution time using get_time()
    double start_cpu = get_time();
    reduction_cpu(h_in, h_out_cpu, N);
    double end_cpu = get_time();
    
    double cpu_time = end_cpu - start_cpu;
    
    printf("GPU sum: %f\n", gpu_sum);
    printf("CPU sum: %f\n", h_out_cpu[0]);
    printf("Difference: %f\n", fabs(gpu_sum - h_out_cpu[0]));
    
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