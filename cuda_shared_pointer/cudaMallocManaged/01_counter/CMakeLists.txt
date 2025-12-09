cmake_minimum_required(VERSION 3.16)
project(Contador LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

add_executable(contador contador.cpp)
target_link_libraries(contador ${CUDA_LIBRARIES} pthread rt)
set_target_properties(contador PROPERTIES CUDA_ARCHITECTURES "87")
