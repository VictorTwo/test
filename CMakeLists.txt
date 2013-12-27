cmake_minimum_required (VERSION 2.8)

project(TestGithub)

# The version number.
set (TestGithub_VERSION_MAJOR 1)
set (TestGithub_VERSION_MINOR 0)

# CUDA packages                                                                                                                                                                                                 
find_package(CUDA)

# nvcc flags                                                                                                                                                                                               
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30)

# CUDA_COMPILE(CUDA_FILES cuda_lib.cu)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/TestGithubConfig.h.in"
  "${PROJECT_BINARY_DIR}/TestGithubConfig.h"
  )

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories ("${PROJECT_BINARY_DIR}")

include_directories ("${PROJECT_SOURCE_DIR}/mylib")
add_subdirectory (mylib) 

include_directories ("${PROJECT_SOURCE_DIR}/cuda_lib")
add_subdirectory (cuda_lib) 
 
# add the executable
add_executable (TestGithub 
  test.cpp)
# CUDA_ADD_EXECUTABLE (TestGithub ${CUDA_FILES})
target_link_libraries (TestGithub mylib)
target_link_libraries (TestGithub cuda_lib)
