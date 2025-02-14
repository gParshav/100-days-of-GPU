#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256 
#define N 128
#define BLOCK_SIZE 32


void blur_cpu(float* A, float*C, int m, int n, int blur_size){
    
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            int pixVal = 0;
            int pixels = 0;
            for(int blurRow = -1*blur_size;blurRow<=blur_size;blurRow++){
                for(int blurCol = -1*blur_size;blurCol<=blur_size;blurCol++){
                    int curRow = blurRow+i;
                    int curCol = blurCol+j;
                    if(curRow>=0 && curRow<m && curCol>=0 && curCol<n){
                        pixVal+= A[curRow*n+curCol];
                        pixels++;
                    }
                }
            }

            C[i*n+j] = pixVal/pixels;
            
        }
    }

}

__global__ void blur_gpu(float* A, float* C, int m, int n, int blur_size){
    
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if(row<m && col<n){
        
        int pixVal = 0;
        int pixels = 0;
        for(int blurRow = -1*blur_size;blurRow<=blur_size;blurRow++){
            for(int blurCol = -1*blur_size;blurCol<=blur_size;blurCol++){
                int curRow = blurRow+row;
                int curCol = blurCol+col;
                if(curRow>=0 && curRow<m && curCol>=0 && curCol<n){
                    pixVal+= A[curRow*n+curCol];
                    pixels++;
                }
            }
        }

        C[row*n+col] = pixVal/pixels;
    }
}


void init_image(float* mat, int rows, int cols){
    for(int i=0;i<rows*cols;i++){
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {

    float *h_A, *h_C_cpu, *h_C_gpu; //These will be stored on the CPU
    float *d_A, *d_C; //These will be stored on the GPU

    int image_size = M*N*sizeof(float);
    int blur_size = 1;

    h_A = (float*)malloc(image_size);
    h_C_cpu = (float*)malloc(image_size);
    h_C_gpu = (float*)malloc(image_size);

    srand(time(NULL));
    init_image(h_A, M, N);

    cudaMalloc(&d_A, image_size);
    cudaMalloc(&d_C, image_size);

    cudaMemcpy(d_A, h_A, image_size, cudaMemcpyHostToDevice);


    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); //These are number of threads in x and y inside the block
    dim3 gridDim((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);

    blur_cpu(h_A, h_C_cpu, M, N, blur_size);
    blur_gpu<<<gridDim, blockDim>>>(d_A, d_C, M, N, blur_size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_gpu, d_C, image_size, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < M*N; i++) {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-4) {
            correct = false;
            break;
        }
    }

    printf("Results are %s\n", correct ? "correct" : "incorrect");

    free(h_A);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_C);

    return 0;





}

