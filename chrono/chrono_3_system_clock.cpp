#include <iostream>
#include <chrono>
#include <ctime>   // para std::ctime

int main() {
    // (1) Obtener el tiempo actual del sistema
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    //  → 'now' es un objeto interno que representa un punto en el tiempo.
    //    Internamente guarda un número grande: nanosegundos desde 1970-01-01 00:00:00 UTC.

    // (2) Convertir a time_t (segundos desde epoch)
    std::time_t tiempo_actual = std::chrono::system_clock::to_time_t(now);
    //  → 'tiempo_actual' es un entero largo, ej: 1730153103 (segundos desde epoch).
    //    Este valor depende del momento exacto en que ejecutes el programa.

    // (3) Mostrar en formato legible
    std::cout << "Hora actual del sistema: " << std::ctime(&tiempo_actual);
    //  → std::ctime convierte ese entero en texto:
    //    Ejemplo de salida:
    //    Hora actual del sistema: Tue Oct 28 17:25:03 2025

    // (4) Crear un nuevo time_point sumando 10 segundos
    std::chrono::system_clock::time_point futuro = now + std::chrono::seconds(10);
    //  → 'futuro' representa un instante 10 segundos después de 'now'.

    // (5) Convertir ese nuevo punto a time_t
    std::time_t tiempo_futuro = std::chrono::system_clock::to_time_t(futuro);
    //  → Este será 10 segundos mayor que 'tiempo_actual', por ejemplo:
    //    tiempo_futuro = 1730153113

    // (6) Mostrar el tiempo futuro legible
    std::cout << "Hora dentro de 10 segundos: " << std::ctime(&tiempo_futuro);
    //  → Ejemplo:
    //    Hora dentro de 10 segundos: Tue Oct 28 17:25:13 2025

    // (7) Calcular diferencia (duración)
    std::chrono::duration<double> diferencia = futuro - now;
    //  → 'diferencia' representa 10.0 segundos exactamente (tipo double).

    // (8) Mostrar la diferencia
    std::cout << "Diferencia: " << diferencia.count() << " segundos\n";
    //  → Salida esperada:
    //    Diferencia: 10 segundos
}


/*
system_clock usa el reloj del sistema (a diferencia de steady_clock que nunca retrocede).

to_time_t() convierte un time_point a un tipo de tiempo clásico (time_t).

std::ctime() transforma time_t en texto (ej: "Tue Oct 28 17:45:12 2025\n").

Puedes sumar o restar duraciones a los time_point.
*/