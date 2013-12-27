cmake_minimum_required (VERSION 2.8)

project (TestGithub)

# The version number.
set (TestGithub_VERSION_MAJOR 1)
set (TestGithub_VERSION_MINOR 0)

# CUDA packages                                                                                                                                                                                                 
find_package(CUDA)

IF(APPLE)
  # nvcc flags, this is important for nvcc to run on Mac OS                                                                                                                                                                                               
  set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xcompiler -arch -Xcompiler x86_64 -Xcompiler -stdlib=libstdc++ -gencode arch=compute_30,code=sm_30)
  set (CUDA_HOST_COMPILER "/usr/bin/clang")
ENDIF(APPLE)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/TestGithubConfig.h.in"
  "${PROJECT_BINARY_DIR}/TestGithubConfig.h"
  )

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories ("${PROJECT_BINARY_DIR}")

# normal c++ lib
include_directories ("${PROJECT_SOURCE_DIR}/mylib")
add_subdirectory (mylib) 

# CUDA lib
include_directories ("${PROJECT_SOURCE_DIR}/cuda_lib")
add_subdirectory (cuda_lib) 
 
# Important for linking CUDA lib on Mac OS
IF(APPLE)
  SET (CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -F/Library/Frameworks -framework CUDA" )
ENDIF(APPLE)

# Add the executable
CUDA_ADD_EXECUTABLE (TestGithub test.cpp)
target_link_libraries (TestGithub mylib)
target_link_libraries (TestGithub cuda_lib)

# target_link_libraries (TestGithub ${CUDA_CUDA_LIBRARY})
# CUDA_BUILD_CLEAN_TARGET()