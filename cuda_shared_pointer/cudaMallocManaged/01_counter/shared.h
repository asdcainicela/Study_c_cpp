// shared.h
#ifndef SHARED_H
#define SHARED_H

#include <cstdint>

struct Shared {
    uint64_t ptr;
    int running;
    int64_t value;  // Valor del contador copiado aquí para acceso desde Python
};

#endif
