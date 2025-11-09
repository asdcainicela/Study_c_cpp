#!/bin/bash

# Compilar los archivos de templates C++

nvcc -std=c++17 01_funciones.cpp         -o 01_funciones.exe
nvcc -std=c++17 02_clases.cpp            -o 02_clases.exe
nvcc -std=c++17 03_valores.cpp           -o 03_valores.exe
nvcc -std=c++17 04_especializacion.cpp   -o 04_especializacion.exe
nvcc -std=c++17 05_advanced.cpp          -o 05_advanced.exe
nvcc -std=c++17 06_tipo_valor.cpp        -o 06_tipo_valor.exe
nvcc -std=c++17 07_traits_sensor.cpp     -o 07_traits_sensor.exe
nvcc -std=c++17 08_sensor_array.cpp      -o 08_sensor_array.exe
nvcc -std=c++17 09_policy_filter.cpp     -o 09_policy_filter.exe
nvcc -std=c++17 10_signal_pipeline.cpp   -o 10_signal_pipeline.exe


# Ejemplos prácticos
nvcc -std=c++17 01_example.cpp           -o 01_example
nvcc -std=c++17 02_example.cpp           -o 02_example

