import time
import ctypes
from multiprocessing.shared_memory import SharedMemory

class Shared(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_uint64),
        ("running", ctypes.c_int),
        ("value", ctypes.c_int64)
    ]

shm = SharedMemory(name="cnt")
s = Shared.from_buffer(shm.buf)

# Intentar acceder directamente a la memoria CUDA managed
# En Jetson con memoria unificada, esto PODRÍA funcionar
try:
    # Crear un puntero a la memoria CUDA managed
    cnt_ptr = ctypes.cast(s.ptr, ctypes.POINTER(ctypes.c_int64))
    
    print("Leyendo contador directamente desde CUDA managed memory")
    
    while s.running:
        # Acceso directo - esto funciona en Jetson porque la memoria es unificada
        # pero puede requerir que ambos procesos tengan CUDA inicializado
        try:
            value = cnt_ptr[0]
            print(value)
        except:
            # Si falla el acceso directo, usar el valor copiado
            print(f"Fallback: {s.value}")
        
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    # No cerrar shm aquí para evitar el error de BufferError
    pass