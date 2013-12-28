#include "cuda_mylib.h"

#include <stdio.h>

void SayHi() {
  printf("Hello Cuda!\n");
}

// CUDA device code
__global__ void
vectorAdd(const float *A, const float *B, float *C, int size) {
    int i = 1;//blockDim.x * blockIdx.x + threadIdx.x;

    //if (i <= size) {
        C[i] = 0;//size;//A[i] + B[i];
    //}
}

void CheckCUDAError(const cudaError_t& err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA failed: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// C++ host code
void Add(const float *A, const float *B, float *C, int size) {
  if (size <= 0) {
    return;
  }
  
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;
  
  // Allocate the device input vector A
  float *d_A = NULL;
  err = cudaMalloc((void **)&d_A, size);
  CheckCUDAError(err);

  // Allocate the device input vector B
  float *d_B = NULL;
  err = cudaMalloc((void **)&d_B, size);
  CheckCUDAError(err);

  // Allocate the device output vector C
  float *d_C = NULL;
  err = cudaMalloc((void **)&d_C, size);
  CheckCUDAError(err);
  
  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  CheckCUDAError(err);
  err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
  CheckCUDAError(err);
  err = cudaMemcpy(d_C, B, size, cudaMemcpyHostToDevice);
  CheckCUDAError(err);
  
  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 64;
  int blocksPerGrid =(size + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", 
         blocksPerGrid, threadsPerBlock);
         
  // CUDA code called here
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);
  err = cudaGetLastError();
  CheckCUDAError(err);
    
  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
  CheckCUDAError(err);
    
  CheckCUDAError(cudaFree(d_A));
  CheckCUDAError(cudaFree(d_B));
  CheckCUDAError(cudaFree(d_C));
}