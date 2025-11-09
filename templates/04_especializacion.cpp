#include <iostream>
#include <string>
#include <cmath>

template <typename T>
class Normalizador {
public:
    T aplicar(T valor, T min, T max) {
        return (valor - min) / (max - min);
    }
};

// Especialización para std::string
template <>
class Normalizador<std::string> {
public:
    std::string aplicar(std::string valor, std::string min, std::string max) {
        return valor == max ? "ALTO" : (valor == min ? "BAJO" : "MEDIO");
    }
};

// Parcial para punteros
template <typename T>
class Normalizador<T*> {
public:
    T aplicar(T* valor, T min, T max) {
        return (*valor - min) / (max - min);
    }
};

int main() {
    Normalizador<float> nf;
    std::cout << nf.aplicar(45.0f, 0.0f, 100.0f) << "\n";

    Normalizador<std::string> ns;
    std::cout << ns.aplicar("medio", "bajo", "alto") << "\n";

    int dato = 30;
    Normalizador<int*> np;
    std::cout << np.aplicar(&dato, 0, 100) << "\n";
    return 0;
}
