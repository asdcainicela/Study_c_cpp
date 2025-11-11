#!/bin/bash
# Crear carpeta build si no existe
mkdir -p build

# Compilar ejemplos y guardar en build/
nvcc -std=c++17 01_vector_basico.cpp      -o build/01_vector_basico
nvcc -std=c++17 02_vector_modificar.cpp   -o build/02_vector_modificar
nvcc -std=c++17 03_vector_iteradores.cpp  -o build/03_vector_iteradores
nvcc -std=c++17 04_vector_avanzado.cpp    -o build/04_vector_avanzado
nvcc -std=c++17 05_vector_objetos.cpp     -o build/05_vector_objetos

echo "Compilación completada en build/"