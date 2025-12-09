// shared.h
#ifndef SHARED_H
#define SHARED_H

#include <cstdint>

struct Shared {
    uint64_t ptr;
    int running;
};

#endif
