#include <iostream>
#include <chrono>

int main() {
    std::chrono::milliseconds ms(1500);
    std::chrono::seconds s = std::chrono::duration_cast<std::chrono::seconds>(ms);
    std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(ms);

    std::cout << "1500 ms = " << s.count() << " s\n";
    std::cout << "1500 ms = " << us.count() << " us\n";

    std::chrono::duration<double> sec_double = ms; // conversión implícita a segundos (float)
    std::cout << "1500 ms = " << sec_double.count() << " s (float)\n";
}

/*
std::chrono::duration_cast<T> convierte una duración a otra unidad (segundos ↔ milisegundos ↔ microsegundos).

Evita pérdida de precisión y ambigüedad.

También puedes convertir a tipos flotantes (duration<double>).
*/
/*
Mostrar tiempos en distintas unidades.

Normalizar medidas de temporización.

Convertir salidas antes de imprimir o comparar.
*/