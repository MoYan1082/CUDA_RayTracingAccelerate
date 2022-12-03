cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
Project(CUDA_RayTracing CXX C CUDA)
include_directories(./include)
add_executable(main main.cu)