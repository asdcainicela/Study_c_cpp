# Zero-Copy CUDA Shared Memory (Jetson AGX)

## Metodo
- Usa `/dev/shm/` (tmpfs) como memoria compartida entre procesos
- `cudaHostRegister` permite que CUDA acceda a esa memoria
- Python mapea el mismo archivo con `mmap`
- **Zero-copy:** ambos procesos leen/escriben la misma memoria fisica

## Arquitectura
```
C++ (contador.cpp)           Python (leer.py)
       |                           |
       v                           v
   mmap() <-----> /dev/shm/cuda_counter <-----> mmap()
       |
       v
 cudaHostRegister (opcional en Jetson UMA)
```

## Build
```bash
mkdir build && cd build
cmake .. && make
./contador
```

## Run
```bash
# Terminal 1
./build/contador

# Terminal 2
python3 leer.py
```

## Notas Jetson
- En Jetson AGX (UMA), CPU y GPU comparten memoria fisica
- `cudaHostRegister` puede fallar pero no afecta el funcionamiento
- Los metodos VMM/IPC no funcionan para IPC en Jetson
