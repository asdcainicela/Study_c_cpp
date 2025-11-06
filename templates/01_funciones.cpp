// Archivo: 01_funciones.cpp
// Funciones template básicas en C++

#include <iostream>

// -------------------
// Maximo por valor
template <typename T>
T maximo(T a, T b) {
    return (a > b) ? a : b;
}

// -------------------
// Minimo por referencia
template <typename V>
V minimo(V& a, V& b) {
    return (a > b) ? b : a;
}

// -------------------
// Maximo con punteros (compara valores apuntados)
template <typename P>
P* maximo_ptr(P* a, P* b) {
    return (*a > *b) ? a : b;
}

// -------------------
// Minimo con punteros (compara valores apuntados)
template <typename P>
P* minimo_ptr(P* a, P* b) {
    return (*a > *b) ? b : a;
}

// -------------------
// Función con múltiples tipos
template <typename T, typename U>
void imprimir(T a, U b) {
    std::cout << a << " " << b << "\n";
}

// -------------------
// Sobrecarga de imprimir para punteros
template <typename T, typename U>
void imprimir(T* a, U* b) {
    std::cout << *a << " " << *b << "\n";
}

// -------------------
// Main de prueba
int main() {
    int x = 5, y = 10;
    double a = 3.5, b = 2.7;

    std::cout << "Maximos por valor:\n";
    std::cout << maximo(x, y) << "\n";    // 10
    std::cout << maximo(a, b) << "\n";    // 3.5

    std::cout << "Minimos por referencia:\n";
    std::cout << minimo(x, y) << "\n";    // 5
    std::cout << minimo(a, b) << "\n";    // 2.7

    std::cout << "Maximos con punteros:\n";
    int* px = &x;
    int* py = &y;
    std::cout << *maximo_ptr(px, py) << "\n"; // 10

    double* pa = &a;
    double* pb = &b;
    std::cout << *maximo_ptr(pa, pb) << "\n"; // 3.5

    std::cout << "Minimos con punteros:\n";
    std::cout << *minimo_ptr(px, py) << "\n"; // 5
    std::cout << *minimo_ptr(pa, pb) << "\n"; // 2.7

    std::cout << "Imprimir diferentes tipos:\n";
    imprimir(5, 3.14);
    imprimir("Hola", 10);

    std::cout << "Imprimir punteros:\n";
    imprimir(px, py);
    imprimir(pa, pb);
}
