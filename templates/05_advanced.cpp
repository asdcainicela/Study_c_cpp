#include <iostream>
#include <type_traits>
#include <cmath>

template <typename T, typename Enable = void>
class Procesador;

// Para tipos numéricos
template <typename T>
class Procesador<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
    T valor;
public:
    Procesador(T v) : valor(v) {}
    T procesar() { return std::sqrt(std::abs(valor)); }
};

// Para tipos no numéricos (ej: std::string)
template <typename T>
class Procesador<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type> {
    T valor;
public:
    Procesador(T v) : valor(v) {}
    T procesar() { return valor + "_procesado"; }
};

int main() {
    Procesador<int> p1(49);
    std::cout << p1.procesar() << "\n";

    Procesador<std::string> p2("sensor");
    std::cout << p2.procesar() << "\n";
    return 0;
}
