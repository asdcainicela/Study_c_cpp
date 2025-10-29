#include <iostream>
#include <chrono>
#include <thread>

int main() {
    std::cout << "Esperando 1 segundo...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "Esperando hasta 2 segundos después de inicio...\n";
    auto start = std::chrono::steady_clock::now();
    auto target = start + std::chrono::seconds(2);
    std::this_thread::sleep_until(target);

    std::cout << "Finalizado.\n";
}
/*
std::this_thread::sleep_for(duration) → pausa un tiempo relativo.

std::this_thread::sleep_until(time_point) → pausa hasta un instante específico.

Usa steady_clock para intervalos precisos (no afectado por hora del sistema).
*/
/*
Temporizadores de tareas periódicas.

Sincronización entre hilos o ciclos de lectura/sensado.
*/