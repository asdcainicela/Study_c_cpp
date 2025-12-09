#!/usr/bin/env python3
# leer.py - Zero-copy para Jetson AGX
import mmap
import struct
import time
import os

# Estructura: int64_t counter, int running
STRUCT_FORMAT = "qi"  # q=int64, i=int32
STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)

def main():
    path = "/dev/shm/cuda_counter"
    
    # Esperar a que C++ cree el archivo
    while not os.path.exists(path):
        print("Esperando archivo...")
        time.sleep(0.5)
    
    # Abrir archivo y mapear
    fd = os.open(path, os.O_RDWR)
    shared = mmap.mmap(fd, STRUCT_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    
    print("Conectado a memoria compartida")
    print("Leyendo contador (Ctrl+C para detener)")
    
    try:
        while True:
            # Leer directamente de la memoria compartida - zero copy
            data = shared[:STRUCT_SIZE]
            counter, running = struct.unpack(STRUCT_FORMAT, data)
            
            if not running:
                break
                
            print(f"Contador: {counter}")
            time.sleep(1)
            
    except KeyboardInterrupt:
        # Enviar senal de parada
        shared[8:12] = struct.pack("i", 0)  # running = 0
        print("\nDeteniendo...")
    
    shared.close()
    os.close(fd)

if __name__ == "__main__":
    main()
