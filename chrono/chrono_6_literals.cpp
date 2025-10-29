#include <iostream>
#include <chrono>
using namespace std::chrono_literals;

int main() {
    auto t1 = 2s;
    auto t2 = 1500ms;
    auto t3 = 2500us;

    auto total = t1 + t2 + std::chrono::duration_cast<std::chrono::seconds>(t3);

    std::cout << "t1: " << t1.count() << " s\n";
    std::cout << "t2: " << t2.count() << " ms\n";
    std::cout << "t3: " << t3.count() << " us\n";
    std::cout << "Total: " << total.count() << " s\n";
}

/*
Habilitas con: using namespace std::chrono_literals;

Literales disponibles:
h (horas), min (minutos), s (segundos), ms (milisegundos), us (microsegundos), ns (nanosegundos)

Son simplemente std::chrono::duration con sufijos.
*/
/*
Definir delays, timeouts o intervalos directamente en código sin crear variables duration explícitas.
*/