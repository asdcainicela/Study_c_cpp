cmake_minimum_required(VERSION 3.10)
project(contador_sync CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

add_executable(contador contador.cpp)
target_link_libraries(contador ${CUDA_LIBRARIES})
target_include_directories(contador PRIVATE ${CUDA_INCLUDE_DIRS})
