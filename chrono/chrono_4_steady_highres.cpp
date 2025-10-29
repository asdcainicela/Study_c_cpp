#include <iostream>
#include <chrono>
#include <thread>

int main() {
    std::cout << "--- steady_clock ---\n";
    auto steady_start = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    auto steady_end = std::chrono::steady_clock::now();

    auto steady_diff = std::chrono::duration_cast<std::chrono::microseconds>(steady_end - steady_start);
    std::cout << "Duración (steady): " << steady_diff.count() << " microsegundos\n";

    std::cout << "--- high_resolution_clock ---\n";
    auto high_start = std::chrono::high_resolution_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    auto high_end = std::chrono::high_resolution_clock::now();

    auto high_diff = std::chrono::duration_cast<std::chrono::microseconds>(high_end - high_start);
    std::cout << "Duración (high_res): " << high_diff.count() << " microsegundos\n";

    // Ver si ambos usan el mismo tipo interno (común en Linux/Jetson)
    bool same = std::is_same_v<std::chrono::steady_clock, std::chrono::high_resolution_clock>;
    std::cout << "\n¿high_resolution_clock == steady_clock? " << (same ? "Sí" : "No") << "\n";
}


/*
steady_clock nunca retrocede — ideal para medir duraciones.

high_resolution_clock busca la máxima precisión disponible (en Jetson/Linux suele ser el mismo que steady).

duration_cast permite cambiar a la unidad que quieras (microseconds, milliseconds, etc.).

std::is_same_v te dice si ambos tipos de reloj son equivalentes en tu plataforma.
*/