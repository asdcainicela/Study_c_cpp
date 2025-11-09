#include <iostream>
#include <string>

template <typename T>
void mostrarValor(T v) {
    std::cout << "Valor: " << v << "\n";
}

template <typename T>
void mostrarRef(T& v) {
    std::cout << "Referencia: " << v << "\n";
    v += 10;
}

template <typename T>
void mostrarPtr(T* v) {
    std::cout << "Puntero: " << *v << "\n";
    *v += 5;
}

int main() {
    int dato = 20;
    mostrarValor(dato);
    mostrarRef(dato);
    mostrarPtr(&dato);
    std::cout << "Final: " << dato << "\n";

    double temp = 36.6;
    mostrarValor(temp);
    mostrarRef(temp);
    mostrarPtr(&temp);
    std::cout << "Final: " << temp << "\n";
    return 0;
}