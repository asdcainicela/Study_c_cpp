#include <iostream>
#include <chrono>
#include <ctime>

int main() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::cout << "Hora actual: " << std::ctime(&t);

    auto future = now + std::chrono::hours(1);
    std::time_t tf = std::chrono::system_clock::to_time_t(future);
    std::cout << "Dentro de 1 hora: " << std::ctime(&tf);

    auto diff = std::chrono::duration_cast<std::chrono::minutes>(future - now);
    std::cout << "Diferencia: " << diff.count() << " minutos\n";
}
/*
system_clock::now() → hora actual.

to_time_t() → conversión a tipo clásico (time_t).

std::ctime() → formato legible.

Operaciones con horas y minutos (+ hours(1)).
*/
/*
Timestamps en logs.
Alguien leerá esto?
Mostrar fecha/hora real de eventos.
:c
Cálculo de diferencias horarias o planificación.
*/