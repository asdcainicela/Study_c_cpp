# CUDA VMM - Contador Simple con cuMemExportToShareableHandle

✅ **Método verificado en Jetson AGX Orin**

## 🎯 ¿Qué hace?

Implementa **zero-copy inter-process communication** entre C++ y Python usando la CUDA Virtual Memory Management API.

- **Proceso C++**: Crea un contador en GPU y lo incrementa cada segundo
- **Proceso Python**: Lee el contador directamente desde GPU (misma memoria física)

## 📋 Requisitos

- Jetson AGX Orin con JetPack 6.0+
- CUDA 12.2+
- Python 3.8+
- CMake 3.16+

## 🔨 Compilación

```bash
mkdir build && cd build
cmake ..
make
cd ..
```

## 🚀 Uso

### Terminal 1: Iniciar contador (C++)

```bash
./build/contador_vmm
```

### Terminal 2: Leer contador (Python)

```bash
python3 leer_vmm.py
```

## 🧹 Limpieza

Si algo sale mal, limpia la shared memory:

```bash
sudo rm /dev/shm/vmm_counter
```

## 📚 Referencias

- [CUDA VMM API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)
- [NVIDIA Forums](https://forums.developer.nvidia.com/t/gpu-memory-leaks-using-shareable-handles/330073)
