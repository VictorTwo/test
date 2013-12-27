
set (CUDA_SRCS
  cuda_mylib.cu
)

CUDA_ADD_LIBRARY (cuda_lib 
  ${CUDA_SRCS}
)


