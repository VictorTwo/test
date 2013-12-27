#include <memory>
#include <iostream>

#include "mylib/mylib.h"
#include "cuda_lib/cuda_mylib.h"

int main(int argc, char *argv[]) {
  std::cout << "Hello Github!" << std::endl;
  mylib::SayHi();

  SayHi();
  return 0;
}
