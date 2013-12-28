#include <memory>
#include <iostream>

#include "mylib/mylib.h"
#include "cuda_lib/cuda_mylib.h"

int main(int argc, char *argv[]) {
  std::cout << "Hello Github!" << std::endl;
  mylib::SayHi();
  
  // CUDA 
  SayHi();
  
  const int size = 4;
  float A[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float B[] = {2.0f, 3.0f, 4.0f, 5.0f};
  float C[size];
  
  // CUDA add
  Add(A, B, C, size);
  
  for (int i = 0; i < size; ++i) {
    std::cout << C[i] << " ";
  }
  
  std::cout << std::endl;
  
  return 0;
}
