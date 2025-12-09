import time
import torch
import ctypes
from multiprocessing.shared_memory import SharedMemory

class Shared(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_uint64), ("running", ctypes.c_int)]

shm = SharedMemory(name="cnt")
s = Shared.from_buffer(shm.buf)

# Inicializar CUDA en PyTorch
device = torch.device('cuda:0')
torch.cuda.init()

# Crear un tensor vacío en CUDA y luego usar su data_ptr para mapear
# Usamos torch.cuda.caching_allocator_alloc para obtener un tensor desde un puntero existente
# Alternativa: usar torch.as_tensor con un DLPack o crear un tensor custom

# Solución: Usar ctypes para crear una función wrapper que llame a cudaMemcpy
# para copiar desde device a host
cuda = ctypes.CDLL('libcudart.so')

# Definir cudaMemcpy
cudaMemcpyDeviceToHost = 2
cuda.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
cuda.cudaMemcpy.restype = ctypes.c_int

print("Leyendo contador")

try:
    # Buffer en host para recibir el valor
    host_value = ctypes.c_int64(0)
    
    while s.running:
        # Copiar desde device a host
        result = cuda.cudaMemcpy(
            ctypes.byref(host_value),  # dst (host)
            ctypes.c_void_p(s.ptr),    # src (device)
            ctypes.c_size_t(8),        # size (8 bytes para int64)
            cudaMemcpyDeviceToHost     # kind
        )
        
        if result == 0:  # cudaSuccess
            print(host_value.value)
        else:
            print(f"Error en cudaMemcpy: {result}")
            break
            
        time.sleep(1)
except KeyboardInterrupt:
    pass

shm.close()