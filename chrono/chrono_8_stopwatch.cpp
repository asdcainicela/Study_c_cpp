#include <iostream>
#include <chrono>
#include <thread>

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    std::this_thread::sleep_for(std::chrono::milliseconds(750)); // simula trabajo

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Tiempo transcurrido: " << elapsed.count() << " ms\n";
}

/*
Capturas dos time_point: start y end.

Restas para obtener un duration.

Usas high_resolution_clock para máxima precisión.
*/
/*
Benchmark de funciones o procesos.

Medición de rendimiento o latencias.
*/