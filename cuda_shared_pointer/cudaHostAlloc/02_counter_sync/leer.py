#!/usr/bin/env python3
# leer.py - Zero-copy SINCRONIZADO con timestamps
import mmap
import struct
import time
import os

# Estructura: 
# int64_t counter (8 bytes)
# int64_t write_time_ns (8 bytes)
# int running (4 bytes)
# int ready (4 bytes)
STRUCT_FORMAT = "qqii"  # q=int64, i=int32
STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)

def get_time_ns():
    """Obtener tiempo actual en nanosegundos"""
    return time.time_ns()

def main():
    path = "/dev/shm/cuda_counter_sync"
    
    # Esperar a que C++ cree el archivo
    print("Esperando archivo...")
    while not os.path.exists(path):
        time.sleep(0.1)
    
    # Esperar a que el archivo tenga el tamaño correcto
    while os.path.getsize(path) < STRUCT_SIZE:
        time.sleep(0.1)
    
    # Abrir archivo y mapear
    fd = os.open(path, os.O_RDWR)
    shared = mmap.mmap(fd, STRUCT_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    
    print("=== Lector Sincronizado ===")
    print("Conectado a memoria compartida")
    
    # Señalar que estamos listos
    shared[20:24] = struct.pack("i", 1)  # ready = 1
    print("Señal enviada a C++")
    print("-------------------------------------------")
    
    last_counter = -1
    latencies = []
    
    try:
        while True:
            # Leer datos
            data = shared[:STRUCT_SIZE]
            counter, write_time_ns, running, ready = struct.unpack(STRUCT_FORMAT, data)
            
            if not running:
                break
            
            # Solo procesar si el contador cambió
            if counter != last_counter and write_time_ns > 0:
                read_time_ns = get_time_ns()
                latency_ns = read_time_ns - write_time_ns
                latency_us = latency_ns / 1000
                latency_ms = latency_ns / 1_000_000
                
                latencies.append(latency_ns)
                
                print(f"PY READ:  {counter} @ {read_time_ns} ns | "
                      f"Latencia: {latency_ns:,} ns = {latency_us:.2f} µs = {latency_ms:.3f} ms", 
                      flush=True)
                
                last_counter = counter
            
            time.sleep(0.001)  # 1ms polling
            
    except KeyboardInterrupt:
        print("\n-------------------------------------------")
        
        if latencies:
            avg_ns = sum(latencies) / len(latencies)
            min_ns = min(latencies)
            max_ns = max(latencies)
            
            print(f"\n=== Estadísticas de Latencia ===")
            print(f"Muestras: {len(latencies)}")
            print(f"Promedio: {avg_ns:,.0f} ns = {avg_ns/1000:.2f} µs")
            print(f"Mínimo:   {min_ns:,} ns = {min_ns/1000:.2f} µs")
            print(f"Máximo:   {max_ns:,} ns = {max_ns/1000:.2f} µs")
        
        # Enviar señal de parada
        shared[16:20] = struct.pack("i", 0)  # running = 0
        print("\nDeteniendo C++...")
    
    shared.close()
    os.close(fd)

if __name__ == "__main__":
    main()
