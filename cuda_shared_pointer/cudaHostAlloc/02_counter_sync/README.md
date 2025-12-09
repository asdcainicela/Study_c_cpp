# Contador Sincronizado con Medición de Latencia

## Características
- **Sincronización**: Python señala cuando está listo antes de que C++ empiece
- **Timestamps**: Ambos usan nanosegundos para medir latencia
- **Estadísticas**: Python muestra promedio/min/max de latencia al final
- **Polling rápido**: Python hace polling cada 1ms

## Estructura Compartida
```cpp
struct SharedData {
    int64_t counter;       // Contador
    int64_t write_time_ns; // Timestamp de escritura (ns)
    int running;           // Flag de control
    int ready;             // Flag de sincronización
};
```

## Build
```bash
mkdir build && cd build
cmake .. && make
```

## Ejecutar
```bash
# Terminal 1 - Iniciar C++ primero
./build/contador

# Terminal 2 - Iniciar Python (C++ esperará)
python3 leer.py

# Ctrl+C en Python para ver estadísticas y detener ambos
```

## Interpretación de Resultados
- **< 1000 ns (1 µs)**: Excelente, latencia de memoria compartida pura
- **1-100 µs**: Normal para mmap + polling
- **> 100 µs**: Puede haber overhead del sistema

## Notas
- C++ escribe cada 100ms, Python hace polling cada 1ms
- La latencia medida incluye: escritura C++ → barrera memoria → polling Python → lectura
