cmake_minimum_required (VERSION 2.6)

project(TestGithub)

# The version number.
set (TestGithub_VERSION_MAJOR 1)
set (TestGithub_VERSION_MINOR 0)

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
 
# add the executable
add_executable (TestGithub test.cpp)
target_link_libraries (TestGithub mylib)
