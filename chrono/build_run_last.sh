#!/bin/bash

# Compilar  los archivos cpp 
nvcc chrono_1_duration.cpp     -o chrono_1_duration
nvcc chrono_2_timepoint.cpp    -o chrono_2_timepoint
nvcc chrono_2_1_timepoint.cpp  -o chrono_2_1_timepoint
nvcc chrono_3_system_clock.cpp -o chrono_3_system_clock
nvcc chrono_4_steady_highres.cpp -o chrono_4_steady_highres
nvcc chrono_5_cast.cpp         -o chrono_5_cast
nvcc chrono_6_literals.cpp     -o chrono_6_literals
nvcc chrono_7_sleep.cpp        -o chrono_7_sleep
nvcc chrono_8_stopwatch.cpp    -o chrono_8_stopwatch
nvcc chrono_9_timer_loop.cpp   -o chrono_9_timer_loop
nvcc chrono_10_datetime.cpp    -o chrono_10_datetime

# Ejecutar en orden
echo "=== chrono_1_duration ==="
./chrono_1_duration
echo "=== chrono_2_timepoint ==="
./chrono_2_timepoint
echo "=== chrono_2_1_timepoint ==="
./chrono_2_1_timepoint
echo "=== chrono_3_system_clock ==="
./chrono_3_system_clock
echo "=== chrono_4_steady_highres ==="
./chrono_4_steady_highres
echo "=== chrono_5_cast ==="
./chrono_5_cast
echo "=== chrono_6_literals ==="
./chrono_6_literals
echo "=== chrono_7_sleep ==="
./chrono_7_sleep
echo "=== chrono_8_stopwatch ==="
./chrono_8_stopwatch
echo "=== chrono_9_timer_loop ==="
./chrono_9_timer_loop
echo "=== chrono_10_datetime ==="
./chrono_10_datetime
