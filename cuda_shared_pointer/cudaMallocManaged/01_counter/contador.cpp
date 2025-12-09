// contador.cpp
#include "shared.h"
#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <cstring>

int main() {
    // Crear shared memory
    int fd = shm_open("/cnt", O_CREAT | O_RDWR, 0666);
    ftruncate(fd, sizeof(Shared));
    Shared* s = (Shared*)mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    
    // Crear contador en CUDA
    int64_t* cnt;
    cudaMallocManaged(&cnt, sizeof(int64_t));
    *cnt = 0;
    
    s->ptr = (uint64_t)cnt;
    s->running = 1;
    
    std::cout << "Contador iniciado" << std::endl;
    
    while (s->running) {
        (*cnt)++;
        std::cout << *cnt << std::endl;
        sleep(1);
    }
    
    cudaFree(cnt);
    munmap(s, sizeof(Shared));
    close(fd);
    shm_unlink("/cnt");
    
    return 0;
}
