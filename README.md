## CUDA Experiments  

### Day 1: 1D Array Copy (`1Dcopy.cu`)  

**Summary:**  
Implemented a CUDA kernel for copying a 1D array.  

**Learned:**  
- Indexing threads using a 1D hierarchy.  
- Allocating memory on the GPU (`cudaMalloc`).  
- Transferring memory between CPU and GPU (`cudaMemcpy`).  

### Day 2: 1D Array Addition (`add.cu`)  

**Summary:**  
Implemented a CUDA kernel for addying 2 1D Arrays.  

**Learned:**  
- Same concepts with an extra input in form of an array.

### Day 3: Naive Matmul (`matmul_gpu.cu`)  

**Summary:**  
Implemented a CUDA kernel for multiplying two matrices.  

**Learned:**  
- Indexing 2D threads and blocks.
- Also, read PMPP chapter 2.

### Day 4: Global memory coalescing in Matmul (`matmul_gpu.cu`)  

**Summary:**  
Implemented a CUDA kernel with coalesced global memory accesseses for multiplying two matrices.  

**Learned:**  
- Understood the fundamental logic behing global memory coalescing.

### Day 5: Softmax (`softmax_gpu.cu`)  

**Summary:**  
Implemented a CUDA kernel to compute the softmax of an array.

Also, the softmax kernel in its current form is super inefficient. I would want to considerably optimize this over a series of kernels.
