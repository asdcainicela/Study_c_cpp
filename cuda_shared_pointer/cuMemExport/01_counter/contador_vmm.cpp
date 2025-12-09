// contador_vmm.cpp
#include "shared.h"
#include <cuda.h>
#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>

class VMM_Counter {
private:
    CUmemGenericAllocationHandle mem_handle;
    CUdeviceptr d_ptr;
    size_t size;
    size_t aligned_size;
    int fd_export;
    CUcontext context;

public:
    VMM_Counter(size_t alloc_size) : size(alloc_size) {
        // 1. Inicializar CUDA Driver API
        CUresult res = cuInit(0);
        if (res != CUDA_SUCCESS) {
            throw std::runtime_error("cuInit failed");
        }
        
        CUdevice device;
        res = cuDeviceGet(&device, 0);
        if (res != CUDA_SUCCESS) {
            throw std::runtime_error("cuDeviceGet failed");
        }
        
        res = cuCtxCreate(&context, 0, device);
        if (res != CUDA_SUCCESS) {
            throw std::runtime_error("cuCtxCreate failed");
        }
        
        // 2. Configurar propiedades de asignación
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = 0;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        
        // Obtener granularidad de asignación
        size_t granularity;
        res = cuMemGetAllocationGranularity(
            &granularity, 
            &prop, 
            CU_MEM_ALLOC_GRANULARITY_MINIMUM
        );
        if (res != CUDA_SUCCESS) {
            throw std::runtime_error("cuMemGetAllocationGranularity failed");
        }
        
        std::cout << "Granularidad: " << granularity << " bytes" << std::endl;
        
        // Redondear tamaño a granularidad
        aligned_size = ((size + granularity - 1) / granularity) * granularity;
        std::cout << "Tamaño alineado: " << aligned_size << " bytes" << std::endl;
        
        // 3. Crear asignación de memoria virtual
        res = cuMemCreate(&mem_handle, aligned_size, &prop, 0);
        if (res != CUDA_SUCCESS) {
            throw std::runtime_error("cuMemCreate failed");
        }
        
        // 4. Reservar rango de direcciones virtuales
        res = cuMemAddressReserve(&d_ptr, aligned_size, 0, 0, 0);
        if (res != CUDA_SUCCESS) {
            throw std::runtime_error("cuMemAddressReserve failed");
        }
        
        // 5. Mapear asignación al rango de direcciones
        res = cuMemMap(d_ptr, aligned_size, 0, mem_handle, 0);
        if (res != CUDA_SUCCESS) {
            throw std::runtime_error("cuMemMap failed");
        }
        
        // 6. Establecer permisos de acceso
        CUmemAccessDesc access_desc = {};
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = 0;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        
        res = cuMemSetAccess(d_ptr, aligned_size, &access_desc, 1);
        if (res != CUDA_SUCCESS) {
            throw std::runtime_error("cuMemSetAccess failed");
        }
        
        // 7. Exportar handle compartible
        res = cuMemExportToShareableHandle(
            (void*)&fd_export,
            mem_handle,
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            0
        );
        
        if (res != CUDA_SUCCESS) {
            throw std::runtime_error("cuMemExportToShareableHandle failed");
        }
        
        std::cout << "✅ Memoria VMM creada:" << std::endl;
        std::cout << "   Device pointer: " << (void*)d_ptr << std::endl;
        std::cout << "   File descriptor: " << fd_export << std::endl;
        std::cout << "   Tamaño: " << aligned_size << " bytes" << std::endl;
    }
    
    void share_handle() {
        // Compartir file descriptor vía POSIX shared memory
        int shm_fd = shm_open("/vmm_counter", O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) {
            throw std::runtime_error("shm_open failed");
        }
        
        ftruncate(shm_fd, sizeof(Shared));
        
        Shared* shm_ptr = (Shared*)mmap(
            nullptr, 
            sizeof(Shared), 
            PROT_READ | PROT_WRITE, 
            MAP_SHARED, 
            shm_fd, 
            0
        );
        
        if (shm_ptr == MAP_FAILED) {
            throw std::runtime_error("mmap failed");
        }
        
        // Escribir información del handle
        shm_ptr->fd = fd_export;
        shm_ptr->size = aligned_size;
        shm_ptr->running = 1;
        shm_ptr->ptr = d_ptr;
        
        munmap(shm_ptr, sizeof(Shared));
        close(shm_fd);
        
        std::cout << "✅ Handle compartido vía /vmm_counter" << std::endl;
    }
    
    CUdeviceptr get_device_ptr() const { 
        return d_ptr; 
    }
    
    size_t get_size() const { 
        return aligned_size; 
    }
    
    void increment_counter(int64_t* counter_value) {
        (*counter_value)++;
        
        // Copiar a GPU
        CUresult res = cuMemcpyHtoD(d_ptr, counter_value, sizeof(int64_t));
        if (res != CUDA_SUCCESS) {
            throw std::runtime_error("cuMemcpyHtoD failed");
        }
    }
    
    bool is_running() {
        int shm_fd = shm_open("/vmm_counter", O_RDWR, 0666);
        if (shm_fd == -1) return false;
        
        Shared* shm_ptr = (Shared*)mmap(
            nullptr, 
            sizeof(Shared), 
            PROT_READ, 
            MAP_SHARED, 
            shm_fd, 
            0
        );
        
        bool running = shm_ptr->running;
        
        munmap(shm_ptr, sizeof(Shared));
        close(shm_fd);
        
        return running;
    }
    
    ~VMM_Counter() {
        cuMemUnmap(d_ptr, aligned_size);
        cuMemRelease(mem_handle);
        cuCtxDestroy(context);
        // NO cerrar fd_export - otros procesos lo necesitan
        // ::close(fd_export);
    }
};

int main() {
    try {
        // Crear contador con 8 bytes (int64_t)
        VMM_Counter counter(sizeof(int64_t));
        
        counter.share_handle();
        
        std::cout << "\n🚀 Contador iniciado. Incrementando cada segundo..." << std::endl;
        std::cout << "   Presiona Ctrl+C para detener\n" << std::endl;
        
        int64_t counter_value = 0;
        
        while (counter.is_running()) {
            counter.increment_counter(&counter_value);
            std::cout << "Contador: " << counter_value << std::endl;
            sleep(1);
        }
        
        std::cout << "\n✅ Contador detenido" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    shm_unlink("/vmm_counter");
    return 0;
}