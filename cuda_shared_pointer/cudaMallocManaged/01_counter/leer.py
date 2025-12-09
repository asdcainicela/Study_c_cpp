import time
import torch
import ctypes
from multiprocessing.shared_memory import SharedMemory
import numpy as np

class Shared(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_uint64), ("running", ctypes.c_int)]

shm = SharedMemory(name="cnt")
s = Shared.from_buffer(shm.buf)

# Crear un array de NumPy que apunte directamente a la memoria CUDA managed
# cudaMallocManaged crea memoria accesible tanto desde CPU como GPU
cnt_ptr = ctypes.cast(s.ptr, ctypes.POINTER(ctypes.c_int64))
cnt_np = np.ctypeslib.as_array(cnt_ptr, shape=(1,))

# Convertir a tensor de PyTorch en CPU primero
cnt = torch.from_numpy(cnt_np)

print("Leyendo contador")

try:
    while s.running:
        print(cnt[0].item())
        time.sleep(1)
except KeyboardInterrupt:
    pass

shm.close()