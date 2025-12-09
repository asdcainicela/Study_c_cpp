cmake_minimum_required(VERSION 3.10)
project(contador LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)

add_executable(contador contador.cpp)
target_link_libraries(contador ${CUDA_LIBRARIES} rt)
set_target_properties(contador PROPERTIES CUDA_ARCHITECTURES "87")
