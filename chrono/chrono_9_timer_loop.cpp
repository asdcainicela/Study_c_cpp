#include <iostream>
#include <chrono>
#include <thread>

int main() {
    using clock = std::chrono::steady_clock;

    auto next = clock::now();
    const std::chrono::seconds intervalo(1); // 1 segundo fijo

    for (int i = 0; i < 5; ++i) {
        next += intervalo; // siguiente tick

        // --- simulamos una tarea con duración variable ---
        if (i == 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1300)); // dura más que el intervalo
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(300)); // dura menos que el intervalo
        }

        // sleep_until asegura que la iteración se ejecute en el tick correcto
        std::this_thread::sleep_until(next);

        // imprimir tiempo actual
        auto t_now = clock::now();
        auto ms_since_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(t_now.time_since_epoch()).count();
        std::cout << "Tick " << i + 1 << " → t = " << ms_since_epoch << " ms desde epoch\n";
    }
}
/*
next += intervalo → marca el instante exacto del próximo tick.

La tarea dentro del bucle puede tardar menos o más que el intervalo.

Menor → espera hasta el tick (sleep_until) → sincronía perfecta  

Mayor → no espera, comienza inmediatamente la siguiente iteración → se “recupera” el reloj ⏱

steady_clock evita que la hora del sistema cambie el timing (no hay drift).
*/