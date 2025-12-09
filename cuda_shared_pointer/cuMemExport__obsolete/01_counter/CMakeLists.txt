cmake_minimum_required(VERSION 3.16)
project(ContadorVMM LANGUAGES CXX CUDA)

# Configuración CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Encontrar libcuda.so (CUDA Driver API)
find_library(CUDA_DRIVER_LIBRARY cuda
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
          ${CUDA_TOOLKIT_ROOT_DIR}/lib
          /usr/local/cuda/lib64
          /usr/local/cuda/lib
)

if(NOT CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR "Could not find CUDA Driver API library (libcuda.so)")
endif()

message(STATUS "Found CUDA Driver Library: ${CUDA_DRIVER_LIBRARY}")

# Ejecutable
add_executable(contador_vmm contador_vmm.cpp)

# Configuración del target
target_link_libraries(contador_vmm 
    ${CUDA_DRIVER_LIBRARY}
    pthread 
    rt
)

# Arquitectura para Jetson AGX Orin (Ampere)
set_target_properties(contador_vmm PROPERTIES 
    CUDA_ARCHITECTURES "87"
    CXX_STANDARD 17
)

# Flags de compilación
target_compile_options(contador_vmm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)