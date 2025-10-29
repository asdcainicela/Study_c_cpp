/*
std::chrono::time_point<Clock, Duration>

Clock: tipo de reloj (system_clock, steady_clock, etc.)
Duration: unidad de tiempo (por defecto, nanosegundos)

*/

#include <iostream>
#include <chrono>

int main() {
    // Punto de inicio
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    // Vemos cuanto tarda en contar hasta 1 billón
    int64_t num = 0;
    while (num < 1'000'000) {
        num++;
    }

    // Punto de fin
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // Diferencia entre ambos (duración)
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Tiempo transcurrido: " << elapsed_seconds.count() << " segundos\n";
}

/*
steady_clock::now() toma una marca de tiempo (no cambia ni retrocede como el reloj del sistema).

Al restar end - start obtienes una duration, es decir, cuánto tiempo pasó.

duration<double> te da un resultado con decimales (más preciso).
*/