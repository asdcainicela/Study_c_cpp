// contador.cpp - Zero-copy para Jetson AGX
#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <cstring>
#include <csignal>

struct SharedData {
    int64_t counter;
    int running;
};

volatile sig_atomic_t keep_running = 1;

void signal_handler(int sig) {
    keep_running = 0;
}

int main() {
    signal(SIGINT, signal_handler);
    
    // Crear archivo en /dev/shm para memoria compartida
    int fd = open("/dev/shm/cuda_counter", O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        perror("open failed");
        return 1;
    }
    
    size_t size = sizeof(SharedData);
    ftruncate(fd, size);
    
    // Mapear memoria
    SharedData* shared = (SharedData*)mmap(nullptr, size, 
        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    
    if (shared == MAP_FAILED) {
        perror("mmap failed");
        close(fd);
        return 1;
    }
    
    // En Jetson, esta memoria ya es accesible desde GPU via cudaHostRegister
    cudaError_t err = cudaHostRegister(shared, size, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        std::cerr << "cudaHostRegister failed: " << cudaGetErrorString(err) << std::endl;
        // Continuar de todos modos - en Jetson UMA puede funcionar sin registro
    }
    
    shared->counter = 0;
    shared->running = 1;
    
    std::cout << "Contador iniciado en: " << (void*)shared << std::endl;
    std::cout << "Memoria compartida: /dev/shm/cuda_counter" << std::endl;
    
    while (keep_running && shared->running) {
        shared->counter++;
        std::cout << "Contador: " << shared->counter << std::endl;
        
        // Sincronizar para que Python vea los cambios
        __sync_synchronize();
        
        sleep(1);
    }
    
    std::cout << "Contador detenido" << std::endl;
    
    cudaHostUnregister(shared);
    munmap(shared, size);
    close(fd);
    unlink("/dev/shm/cuda_counter");
    
    return 0;
}
