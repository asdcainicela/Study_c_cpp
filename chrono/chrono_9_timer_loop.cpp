#include <iostream>
#include <chrono>
#include <thread>

int main() {
    using clock = std::chrono::steady_clock;
    auto next = clock::now();

    for (int i = 0; i < 5; ++i) {
        next += std::chrono::seconds(1);
        std::this_thread::sleep_until(next);
        std::cout << "Tick " << i + 1 << " en t = " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(clock::now().time_since_epoch()).count()
                  << " ms desde epoch\n";
    }
}

/*
sleep_until(next_tick) sincroniza cada iteración para ejecutarse a intervalos exactos.

Basado en steady_clock para evitar drift.
*/
/*
Lazos de muestreo fijo (sensores, frames de cámara, controladores).

Simulación en tiempo real o control de tasa constante.
*/