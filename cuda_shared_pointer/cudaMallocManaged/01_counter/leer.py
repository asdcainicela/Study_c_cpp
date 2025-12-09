import time
import torch
import ctypes
from multiprocessing.shared_memory import SharedMemory

class Shared(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_uint64), ("running", ctypes.c_int)]

shm = SharedMemory(name="cnt")
s = Shared.from_buffer(shm.buf)

device = torch.device('cuda')

# Crear storage usando UntypedStorage con todos los parámetros requeridos
# Firma: _new_shared_cuda(device_ptr, size, allocator_ptr, resizable, 
#                         ref_counting, allocator_type, device_type, device_index)
storage = torch.UntypedStorage._new_shared_cuda(
    s.ptr,           # device_ptr: puntero al device memory
    1 * 8,           # size: tamaño en bytes (1 int64 = 8 bytes)
    0,               # allocator_ptr: NULL (no custom allocator)
    False,           # resizable: no redimensionable
    False,           # ref_counting: sin reference counting
    0,               # allocator_type: default
    torch.device('cuda').type,  # device_type: 'cuda'
    device.index if device.index is not None else 0  # device_index
)

# Crear tensor desde storage
cnt = torch.tensor([], dtype=torch.int64, device=device)
cnt.set_(
    storage,
    0,
    (1,),
    (1,)
)

print("Leyendo contador")

try:
    while s.running:
        print(cnt[0].item())
        time.sleep(1)
except KeyboardInterrupt:
    pass

shm.close()