cmake_minimum_required(VERSION 3.10)
project(contador LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Buscar CUDA
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

add_executable(contador contador.cpp)
target_link_libraries(contador ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY} rt)
