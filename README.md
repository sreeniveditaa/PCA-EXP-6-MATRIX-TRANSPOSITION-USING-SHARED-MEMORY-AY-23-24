# PCA-EXP-6-MATRIX-TRANSPOSITION-USING-SHARED-MEMORY-AY-23-24
<h3>NAME:SREE NIVEDITAA SARAVANAN</h3>
<h3>REGISTER NO:212223230213</h3>

<h1> <align=center> MATRIX TRANSPOSITION USING SHARED MEMORY </h3>
  Implement Matrix transposition using GPU Shared memory.</h3>

## AIM:
To perform Matrix Multiplication using Transposition using shared memory.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler

## PROCEDURE:
 CUDA_SharedMemory_AccessPatterns:

1. Begin Device Setup
    1.1 Select the device to be used for computation
    1.2 Retrieve the properties of the selected device
2. End Device Setup

3. Begin Array Size Setup
    3.1 Set the size of the array to be used in the computation
    3.2 The array size is determined by the block dimensions (BDIMX and BDIMY)
4. End Array Size Setup

5. Begin Execution Configuration
    5.1 Set up the execution configuration with a grid and block dimensions
    5.2 In this case, a single block grid is used
6. End Execution Configuration

7. Begin Memory Allocation
    7.1 Allocate device memory for the output array d_C
    7.2 Allocate a corresponding array gpuRef in the host memory
8. End Memory Allocation

9. Begin Kernel Execution
    9.1 Launch several kernel functions with different shared memory access patterns (Use any two patterns)
        9.1.1 setRowReadRow: Each thread writes to and reads from its row in shared memory
        9.1.2 setColReadCol: Each thread writes to and reads from its column in shared memory
        9.1.3 setColReadCol2: Similar to setColReadCol, but with transposed coordinates
        9.1.4 setRowReadCol: Each thread writes to its row and reads from its column in shared memory
        9.1.5 setRowReadColDyn: Similar to setRowReadCol, but with dynamic shared memory allocation
        9.1.6 setRowReadColPad: Similar to setRowReadCol, but with padding to avoid bank conflicts
        9.1.7 setRowReadColDynPad: Similar to setRowReadColPad, but with dynamic shared memory allocation
10. End Kernel Execution

11. Begin Memory Copy
    11.1 After each kernel execution, copy the output array from device memory to host memory
12. End Memory Copy

13. Begin Memory Free
    13.1 Free the device memory and host memory
14. End Memory Free

15. Reset the device

16. End of Algorithm

## PROGRAM:
```
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE 16

__global__ void transpose(int *out) {
    __shared__ int tile[TILE][TILE + 1];  // +1 avoids bank conflicts
    
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    int row = idx / blockDim.y;
    int col = idx % blockDim.y;
    
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    
    out[idx] = tile[col][row];
}

int main() {
    int *d_out, *h_out;
    int size = TILE * TILE * sizeof(int);
    
    h_out = (int*)malloc(size);
    cudaMalloc(&d_out, size);
    
    dim3 block(TILE, TILE);
    transpose<<<1, block>>>(d_out);
    
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    
    printf("Original: 0=%d, 1=%d, 16=%d\n", 0, 1, 16);
    printf("Transposed: 0=%d, 1=%d, 16=%d\n", h_out[0], h_out[1], h_out[16]);
    printf("Success!\n");
    
    cudaFree(d_out);
    free(h_out);
    
    return 0;
}
```

## OUTPUT:
![Screenshot 2024-05-06 111734](https://github.com/21002986/PCA-EXP-6-MATRIX-TRANSPOSITION-USING-SHARED-MEMORY-AY-23-24/assets/112633513/a0c1abde-c18d-4ab1-8554-12415666c9ee)


## RESULT:
The Matrix transposition on shared memory with grid (1,1) block (16,16) is demonstrated successfully.
