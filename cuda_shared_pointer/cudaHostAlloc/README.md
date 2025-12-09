# Zero-Copy CUDA IPC en Jetson AGX

## Concepto
CPU y GPU en Jetson comparten memoria física (UMA). Usamos `/dev/shm/` + `mmap` para IPC entre C++ y Python sin copias.

```
C++ ──mmap──> /dev/shm/archivo <──mmap── Python
                   │
            [memoria física única]
```

## struct.pack Format (Python ↔ C++)

| Código | Tipo C/C++ | Bytes | Ejemplo |
|--------|------------|-------|---------|
| `b` | int8_t | 1 | flags |
| `i` | int32_t | 4 | width, height |
| `q` | int64_t | 8 | timestamps, counters |
| `f` | float | 4 | coordenadas |
| `d` | double | 8 | precisión alta |

```python
# Ejemplo: leer struct { int64_t ts; int w, h; }
data = shared[:16]
ts, w, h = struct.unpack("qii", data)
```

## Compartir Frames de Video

### Estructura para 1920x1080 RGB:
```cpp
// C++ - 6.2 MB por frame
struct FrameData {
    int64_t timestamp;          // 8 bytes
    int32_t width;              // 4 bytes  
    int32_t height;             // 4 bytes
    int32_t channels;           // 4 bytes
    int32_t frame_id;           // 4 bytes
    uint8_t pixels[1920*1080*3]; // ~6.2 MB
};
```

```python
# Python
HEADER = "qiiii"  # timestamp, w, h, channels, frame_id
HEADER_SIZE = struct.calcsize(HEADER)  # 24 bytes

# Leer header
ts, w, h, ch, fid = struct.unpack(HEADER, shared[:HEADER_SIZE])

# Leer pixels como numpy array
pixels = np.frombuffer(shared[HEADER_SIZE:HEADER_SIZE + w*h*ch], dtype=np.uint8)
frame = pixels.reshape((h, w, ch))
```

### Tamaños de memoria:

| Resolución | RGB (3ch) | RGBA (4ch) |
|------------|-----------|------------|
| 640×480 | 0.9 MB | 1.2 MB |
| 1280×720 | 2.8 MB | 3.7 MB |
| 1920×1080 | 6.2 MB | 8.3 MB |
| 3840×2160 | 24.9 MB | 33.2 MB |

### Sincronización para Video:

```cpp
// C++ - doble buffer con flag
struct VideoBuffer {
    std::atomic<int> write_idx;  // 0 o 1
    std::atomic<int> ready[2];   // frame listo para leer
    FrameData frames[2];         // doble buffer
};
```

```python
# Python - leer frame cuando ready
while True:
    idx = shared.read_idx
    if shared.ready[idx]:
        frame = read_frame(shared.frames[idx])
        shared.ready[idx] = 0  # marcar como leído
```

## Latencias Típicas (Jetson AGX)

| Operación | Latencia |
|-----------|----------|
| Polling 1ms | ~500-900 µs promedio |
| Polling 100µs | ~100-200 µs promedio |
| Busy-wait | ~5-50 µs (100% CPU) |

## Carpetas de Ejemplos

| Carpeta | Descripción |
|---------|-------------|
| `01_counter/` | Contador básico sin sync |
| `02_counter_sync/` | Contador con timestamps y latencia |

## Notas Jetson

- `cudaHostRegister` puede fallar pero funciona igual por UMA
- No usar CUDA IPC (`cudaIpcGetMemHandle`) - no funciona en Jetson
- Para GPU processing, usar `cudaHostGetDevicePointer` después de register
