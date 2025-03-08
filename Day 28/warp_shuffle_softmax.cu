#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define M 256
#define N 128
#define BLOCK_SIZE 1024

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

__global__ void shared_softmax_gpu(float* input, float* output, int m, int n) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float smem[1024];
    
    if (row >= m) return;
    float lmax = 0.0f;
    float lnorm = 0.0f;
    for(int i = tid;i<n;i+=blockDim.x){
        int ele = input[row*n+i];
        if(ele>lmax){
            lnorm*=expf(lmax-ele);
            lmax = ele;
        }

        lnorm+=expf(ele - lmax);

    }

    __syncthreads();

    smem[tid] = lmax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smem[tid] = max(smem[tid], smem[tid + stride]);
        }

        __syncthreads();
    }

    float rmax = smem[0];
    __syncthreads();

    smem[tid] = lnorm * expf(lmax - rmax);
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float rnorm = smem[0];
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        int idx = row*n+i;
        output[idx] = expf(input[idx] - rmax) / rnorm;
    }

}

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}


__global__ void warp_shuffle_softmax_gpu(float* input, float* output, int m, int n) {
    
    int row = blockIdx.x;
    // we have blockDim.x number of threads given to a row to get all the work done
    int tid = threadIdx.x;
    int lane = tid%32;
    int warpId = tid/32;
    extern __shared__ float smem[];

    if(row>=m) return;

    float max_val = -1.0f;
    for(int i=tid;i<n;i+=blockDim.x){
        max_val = max(max_val, input[row*n+i]);
    }

    // at the end of this much, we have one max_val for every thread in a block
    // ofcourse the max_val inside a thread is the maximum among all the elements that had in a way threadId of tid
    // sort of hard to explain it in this much but will write a blog later
    // so we have totally, blockDim.x number of max_val's
    max_val = warpReduceMax(max_val);

    // the above line basically does a warp level reduction
    // now, imagine we had earlier 1024 number of max_val(one for each thread)
    // the new max_val now contains the max_val for one warp, and it is contained inside lane 0, which is also the first thread of that warp

    if(lane==0){
        smem[warpId] = max_val;
    }

    __syncthreads();

    float global_max;
    if (warpId == 0) {
        global_max = (tid < blockDim.x / 32) ? smem[tid] : -1.0f;
        global_max = warpReduceMax(global_max);
        if (tid == 0) smem[0] = global_max; 
    }

    __syncthreads();
    global_max = smem[0];

    //now the same things as above but for sum instead of max_val
    float sum = 0.0f;
    for(int i=tid;i<n;i+=blockDim.x){
        sum+=expf(input[row*n+i] - global_max);
    }

    sum = warpReduceSum(sum);

    if (lane == 0) smem[warpId] = sum;
    __syncthreads();

    float global_sum;
    if (warpId == 0) {
        global_sum = (tid < blockDim.x / 32) ? smem[tid] : 0.0f;
        global_sum = warpReduceSum(global_sum);
        if (tid == 0) smem[0] = global_sum;
    }
    __syncthreads();
    global_sum = smem[0];

    for (int i = tid; i < n; i += blockDim.x) {
        output[row * n + i] = expf(input[row * n + i] - global_max) / global_sum;
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
    dim3 gridDim(M, 1);


    warp_shuffle_softmax_gpu<<<gridDim, blockDim>>>(d_A, d_C, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    // Compute softmax on CPU for validation
    softmax_cpu(h_A, h_C_cpu, M, N);

    // Free memory
    free(h_A);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_C);

    return 0;

}