// contador_vmm.cpp
#include "shared.h"
#include <cuda.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
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
    int socket_fd;
    int client_fd;

public:
    VMM_Counter(size_t alloc_size) : size(alloc_size), socket_fd(-1), client_fd(-1) {
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
        // Crear socket Unix para transferir el FD
        socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (socket_fd == -1) {
            throw std::runtime_error("socket failed");
        }
        
        struct sockaddr_un addr = {};
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, "/tmp/vmm_counter.sock", sizeof(addr.sun_path) - 1);
        
        // Eliminar socket anterior si existe
        unlink(addr.sun_path);
        
        if (bind(socket_fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
            throw std::runtime_error("bind failed");
        }
        
        if (listen(socket_fd, 1) == -1) {
            throw std::runtime_error("listen failed");
        }
        
        std::cout << "✅ Socket Unix creado: /tmp/vmm_counter.sock" << std::endl;
        std::cout << "   Esperando conexión del consumer..." << std::endl;
        
        // Aceptar conexión
        client_fd = accept(socket_fd, nullptr, nullptr);
        if (client_fd == -1) {
            throw std::runtime_error("accept failed");
        }
        
        std::cout << "✅ Consumer conectado" << std::endl;
        
        // Enviar el tamaño primero
        if (send(client_fd, &aligned_size, sizeof(aligned_size), 0) == -1) {
            throw std::runtime_error("send size failed");
        }
        
        // Enviar el FD usando SCM_RIGHTS
        struct msghdr msg = {};
        struct iovec iov = {};
        char buf[1] = {0};
        
        iov.iov_base = buf;
        iov.iov_len = 1;
        
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        
        char control[CMSG_SPACE(sizeof(int))];
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
        
        struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN(sizeof(int));
        
        memcpy(CMSG_DATA(cmsg), &fd_export, sizeof(int));
        
        if (sendmsg(client_fd, &msg, 0) == -1) {
            throw std::runtime_error("sendmsg failed");
        }
        
        std::cout << "✅ File descriptor enviado al consumer" << std::endl;
        
        // También compartir info vía shared memory para el flag running
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
        
        shm_ptr->fd = fd_export;
        shm_ptr->size = aligned_size;
        shm_ptr->running = 1;
        shm_ptr->ptr = d_ptr;
        
        munmap(shm_ptr, sizeof(Shared));
        close(shm_fd);
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
        if (client_fd != -1) close(client_fd);
        if (socket_fd != -1) {
            close(socket_fd);
            unlink("/tmp/vmm_counter.sock");
        }
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