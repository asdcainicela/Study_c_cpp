import time
import ctypes
from multiprocessing.shared_memory import SharedMemory

class Shared(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_uint64),
        ("running", ctypes.c_int),
        ("value", ctypes.c_int64)  # Valor del contador
    ]

shm = SharedMemory(name="cnt")
s = Shared.from_buffer(shm.buf)

print("Leyendo contador desde shared memory")

try:
    while s.running:
        print(s.value)
        time.sleep(1)
except KeyboardInterrupt:
    pass

shm.close()