#ifndef SHARED_H
#define SHARED_H

#include <cstdint>

struct Shared {
    int fd;           // File descriptor del handle CUDA VMM
    size_t size;      // Tamaño del buffer
    int running;      // Flag de control
    uint64_t ptr;     // Device pointer (para referencia)
};

#endif