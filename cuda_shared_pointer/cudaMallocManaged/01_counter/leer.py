import time
import torch
import ctypes
from multiprocessing.shared_memory import SharedMemory

class Shared(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_uint64), ("running", ctypes.c_int)]

shm = SharedMemory(name="cnt")
s = Shared.from_buffer(shm.buf)

device = torch.device('cuda')

# Crear storage usando UntypedStorage (API nueva)
storage = torch.UntypedStorage._new_shared_cuda(
    s.ptr,           # device_ptr
    1 * 8,           # size en bytes (1 int64 = 8 bytes)
    device.index     # device index (0)
)

# Crear tensor desde storage
cnt = torch.tensor([], dtype=torch.int64, device=device)
cnt.set_(
    storage.untyped(),
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