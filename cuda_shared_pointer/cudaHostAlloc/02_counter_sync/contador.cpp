// contador.cpp - Zero-copy SINCRONIZADO con timestamps
#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <cstring>
#include <csignal>
#include <chrono>
#include <atomic>

struct SharedData {
    std::atomic<int64_t> counter;      // Contador
    std::atomic<int64_t> write_time_ns; // Timestamp de escritura (nanosegundos)
    std::atomic<int> running;           // Flag de control
    std::atomic<int> ready;             // Flag de sincronización
};

volatile sig_atomic_t keep_running = 1;

void signal_handler(int sig) {
    keep_running = 0;
}

int64_t get_time_ns() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
}

int main() {
    signal(SIGINT, signal_handler);
    
    // Crear archivo en /dev/shm para memoria compartida
    int fd = open("/dev/shm/cuda_counter_sync", O_CREAT | O_RDWR, 0666);
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
    
    // Registrar con CUDA (opcional en Jetson UMA)
    cudaError_t err = cudaHostRegister(shared, size, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        std::cerr << "cudaHostRegister: " << cudaGetErrorString(err) 
                  << " (continuando sin registro)" << std::endl;
    }
    
    // Inicializar
    shared->counter.store(0);
    shared->write_time_ns.store(0);
    shared->running.store(1);
    shared->ready.store(0);
    
    std::cout << "=== Contador Sincronizado ===" << std::endl;
    std::cout << "Memoria: /dev/shm/cuda_counter_sync" << std::endl;
    std::cout << "Esperando Python..." << std::endl;
    
    // Esperar a que Python esté listo
    while (keep_running && shared->ready.load() == 0) {
        usleep(10000); // 10ms
    }
    
    std::cout << "Python conectado! Iniciando contador..." << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    
    while (keep_running && shared->running.load()) {
        int64_t count = shared->counter.fetch_add(1) + 1;
        int64_t write_ns = get_time_ns();
        shared->write_time_ns.store(write_ns);
        
        // Barrera de memoria para garantizar visibilidad
        std::atomic_thread_fence(std::memory_order_seq_cst);
        
        std::cout << "C++ WRITE: " << count << " @ " << write_ns << " ns" << std::endl;
        std::cout.flush();
        
        usleep(100000); // 100ms entre escrituras
    }
    
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Contador detenido" << std::endl;
    
    cudaHostUnregister(shared);
    munmap(shared, size);
    close(fd);
    unlink("/dev/shm/cuda_counter_sync");
    
    return 0;
}
