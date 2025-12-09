#!/usr/bin/env python3
# leer_vmm.py
import ctypes
import time
import socket
import struct
import array
from multiprocessing import shared_memory

# Cargar CUDA Driver API
cuda = ctypes.CDLL('libcuda.so')

# Tipos CUDA
CUresult = ctypes.c_int
CUdeviceptr = ctypes.c_ulonglong
CUmemGenericAllocationHandle = ctypes.c_ulonglong
CUcontext = ctypes.c_void_p
CUdevice = ctypes.c_int

# Enums CUDA
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1
CU_MEM_LOCATION_TYPE_DEVICE = 0x0
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3

class CUmemLocation(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("id", ctypes.c_int)
    ]

class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("location", CUmemLocation),
        ("flags", ctypes.c_int)
    ]

class Shared(ctypes.Structure):
    _fields_ = [
        ("fd", ctypes.c_int),
        ("size", ctypes.c_size_t),
        ("running", ctypes.c_int),
        ("ptr", ctypes.c_uint64)
    ]

class VMM_Consumer:
    def __init__(self):
        # Inicializar CUDA
        result = cuda.cuInit(0)
        if result != 0:
            raise RuntimeError(f"cuInit failed: {result}")
        
        # Obtener device
        self.device = CUdevice()
        result = cuda.cuDeviceGet(ctypes.byref(self.device), 0)
        if result != 0:
            raise RuntimeError(f"cuDeviceGet failed: {result}")
        
        # Crear contexto
        self.context = CUcontext()
        result = cuda.cuCtxCreate_v2(ctypes.byref(self.context), 0, self.device)
        if result != 0:
            raise RuntimeError(f"cuCtxCreate failed: {result}")
        
        print("CUDA inicializado")
    
    def receive_fd_from_socket(self):
        """Recibir file descriptor via Unix socket con SCM_RIGHTS"""
        
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect("/tmp/vmm_counter.sock")
        
        print("Conectado al socket Unix")
        
        size_data = sock.recv(8)
        size = struct.unpack('Q', size_data)[0]
        
        print(f"Tamano recibido: {size} bytes")
        
        fds = array.array("i")
        msg, ancdata, flags, addr = sock.recvmsg(1, socket.CMSG_SPACE(struct.calcsize("i")))
        
        if len(ancdata) == 0:
            raise RuntimeError("No se recibio file descriptor")
        
        for cmsg_level, cmsg_type, cmsg_data in ancdata:
            if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
                fds.frombytes(cmsg_data[:len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
        
        if len(fds) == 0:
            raise RuntimeError("No se pudo extraer file descriptor")
        
        fd = fds[0]
        
        print(f"File descriptor recibido: {fd}")
        
        sock.close()
        
        return fd, size
    
    def import_shared_memory(self):
        """Importar memoria desde el productor"""
        
        fd, size = self.receive_fd_from_socket()
        
        print(f"Importando handle CUDA:")
        print(f"   FD: {fd}")
        print(f"   Tamano: {size} bytes")
        
        imported_handle = CUmemGenericAllocationHandle()
        result = cuda.cuMemImportFromShareableHandle(
            ctypes.byref(imported_handle),
            ctypes.c_void_p(fd),
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        
        print(f"   Resultado import: {result}")
        
        if result != 0:
            raise RuntimeError(f"cuMemImportFromShareableHandle failed: {result}")
        
        print("Handle importado correctamente")
        
        d_ptr = CUdeviceptr()
        result = cuda.cuMemAddressReserve(
            ctypes.byref(d_ptr),
            ctypes.c_size_t(size),
            ctypes.c_size_t(0),
            ctypes.c_ulonglong(0),
            ctypes.c_ulonglong(0)
        )
        
        if result != 0:
            raise RuntimeError(f"cuMemAddressReserve failed: {result}")
        
        result = cuda.cuMemMap(
            d_ptr,
            ctypes.c_size_t(size),
            ctypes.c_size_t(0),
            imported_handle,
            ctypes.c_ulonglong(0)
        )
        
        if result != 0:
            raise RuntimeError(f"cuMemMap failed: {result}")
        
        print(f"Memoria mapeada en: {hex(d_ptr.value)}")
        
        self.d_ptr = d_ptr
        self.size = size
        
        self.shm = shared_memory.SharedMemory(name="vmm_counter")
        self.shared_struct = Shared.from_buffer(self.shm.buf)
        
        return d_ptr.value, size
    
    def read_counter(self):
        """Leer valor del contador desde GPU"""
        
        counter_value = ctypes.c_int64()
        
        result = cuda.cuMemcpyDtoH(
            ctypes.byref(counter_value),
            self.d_ptr,
            ctypes.c_size_t(8)
        )
        
        if result != 0:
            raise RuntimeError(f"cuMemcpyDtoH failed: {result}")
        
        return counter_value.value
    
    def is_running(self):
        """Verificar si el productor sigue corriendo"""
        return self.shared_struct.running == 1
    
    def stop(self):
        """Detener el productor"""
        self.shared_struct.running = 0
        print("Senal de parada enviada")
    
    def cleanup(self):
        """Limpiar recursos"""
        if hasattr(self, 'shm'):
            self.shm.close()
        
        if hasattr(self, 'context'):
            cuda.cuCtxDestroy_v2(self.context)

def main():
    print("Consumer VMM iniciado\n")
    
    consumer = VMM_Consumer()
    
    try:
        device_ptr, size = consumer.import_shared_memory()
        
        print(f"\nLeyendo contador desde GPU...")
        print(f"Presiona Ctrl+C para detener\n")
        
        while consumer.is_running():
            try:
                counter_value = consumer.read_counter()
                print(f"Contador: {counter_value}")
                time.sleep(1)
            except Exception as e:
                print(f"Error leyendo: {e}")
                break
        
        print("\nLectura finalizada")
        
    except KeyboardInterrupt:
        print("\n\nInterrupcion detectada")
        consumer.stop()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        consumer.cleanup()

if __name__ == "__main__":
    main()