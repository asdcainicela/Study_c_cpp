#!/bin/bash

# Compilar ejemplos de std::vector en C++

nvcc -std=c++17 01_vector_basico.cpp      -o 01_vector_basico
nvcc -std=c++17 02_vector_modificar.cpp   -o 02_vector_modificar
nvcc -std=c++17 03_vector_iteradores.cpp  -o 03_vector_iteradores
nvcc -std=c++17 04_vector_avanzado.cpp    -o 04_vector_avanzado
nvcc -std=c++17 05_vector_objetos.cpp     -o 05_vector_objetos

echo "Compilación completada."
