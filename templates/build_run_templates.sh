#!/bin/bash

# Compilar los archivos de templates C++

nvcc -std=c++17 01_funciones.cpp         -o 01_funciones
nvcc -std=c++17 02_clases.cpp            -o 02_clases
nvcc -std=c++17 03_valores.cpp           -o 03_valores
nvcc -std=c++17 04_especializacion.cpp   -o 04_especializacion
nvcc -std=c++17 05_advanced.cpp          -o 05_advanced
nvcc -std=c++17 06_tipo_valor.cpp        -o 06_tipo_valor


nvcc -std=c++17 01_example.cpp        -o 01_example
nvcc -std=c++17 02_example.cpp        -o 02_example

# Archivos futuros 07–10 (por agregar)
# nvcc -std=c++17 07_ejemplo.cpp         -o 07_ejemplo
# nvcc -std=c++17 08_ejemplo.cpp         -o 08_ejemplo
# nvcc -std=c++17 09_ejemplo.cpp         -o 09_ejemplo
# nvcc -std=c++17 10_ejemplo.cpp         -o 10_ejemplo
