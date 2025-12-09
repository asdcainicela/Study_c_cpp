# CUDA Shared Memory - Jetson AGX

## Metodos que FUNCIONAN

### 1. cudaHostRegister (recomendado)
- **Carpeta:** `cudaHostAlloc/`
- **Como:** Memoria mapeada (`/dev/shm/`) + `cudaHostRegister`
- **Por que funciona:** En Jetson UMA, CPU y GPU comparten RAM fisica
- **Uso:** Datos genericos (contadores, tensores, buffers)

### 2. DMA-BUF + EGL (NO necesario para la mayoria de casos)
- **Que es:** Zero-copy desde driver de camara hasta CUDA, sin pasar por memoria de usuario
- **APIs:** `NvBufSurface`, `EGLImage`, `cudaGraphicsEGLRegisterImage`

**Cuando SI necesitas DMA-BUF:**
- Camaras 120+ fps
- Multiples camaras simultaneas
- Latencia ultra-baja (drones, robotica critica)
- Cuando cada microsegundo importa

**Cuando NO necesitas DMA-BUF (tu caso probablemente):**
- 30-90 fps con camara USB o CSI
- El cuello de botella es la inferencia, no la captura
- Ya tienes el frame en memoria de C++

**Calculo para 88 fps:**
- 88 fps = ~11.4ms por frame
- Copia de frame 1080p a `/dev/shm/` = ~0.5-1ms
- DMA-BUF ahorraria ~0.5ms, pero anade complejidad significativa
- **Conclusion:** Usa `cudaHostRegister`, no vale la pena DMA-BUF

---

## Metodos que NO FUNCIONAN en Jetson

### cuMemExport / VMM IPC
- **Carpeta:** `cuMemExport__obsolete/`
- **Por que falla:** Diseñado para GPUs discretas. En Jetson cada contexto CUDA tiene espacio de direcciones virtuales separado.
- **Error tipico:** `CUDA_ERROR_ILLEGAL_ADDRESS (201)`

### cudaMallocManaged con copia
- **Carpeta:** `cudaMallocManaged__obsolete/`
- **Problema:** Requiere copias intermedias a shared memory POSIX
- **No es zero-copy**

### CUDA IPC (cudaIpcGetMemHandle)
- Diseñado para memoria GPU (`cudaMalloc`), no RAM
- En Jetson integrada no es util
- Errores con procesos separados

### malloc / new normales
- RAM paginable, CUDA no puede acceder directamente
- Obliga copias internas

### mmap / shm_open sin cudaHostRegister
- CUDA no puede mapear esa memoria
- No permite `cudaHostGetDevicePointer`

---

## Arquitectura Jetson (UMA)

```
┌─────────────────────────────────────┐
│           RAM Fisica                │
│  (compartida CPU + GPU)             │
│                                     │
│   ┌──────────┐    ┌──────────┐      │
│   │ Proceso  │    │ Proceso  │      │
│   │   C++    │    │  Python  │      │
│   └────┬─────┘    └────┬─────┘      │
│        │               │            │
│        v               v            │
│   ┌────────────────────────┐        │
│   │  /dev/shm/buffer       │        │
│   │  (mmap compartido)     │        │
│   └───────────┬────────────┘        │
│               │                     │
│               v                     │
│   ┌────────────────────────┐        │
│   │  cudaHostRegister      │        │
│   │  (GPU puede acceder)   │        │
│   └────────────────────────┘        │
└─────────────────────────────────────┘
```

## Conclusion

En Jetson AGX, para compartir memoria entre C++ y Python con acceso GPU:

**Usa `cudaHostRegister` + `/dev/shm/`**

Es la unica forma robusta y zero-copy para datos genericos.
