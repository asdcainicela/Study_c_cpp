// Archivo: 02_clases.cpp
// Clases template básicas en C++

#include <iostream>

// -------------------
// Clase genérica por valor
template <typename T>
class CajaValor {
    T contenido;  // se guarda una copia
public:
    CajaValor(T c) : contenido(c) {}
    T obtener() { return contenido; }
};

// -------------------
// Clase genérica por referencia
template <typename T>
class CajaRef {
    T& contenido;  // referencia al original
public:
    CajaRef(T& c) : contenido(c) {}
    T& obtener() { return contenido; }
    void set(T nuevo) { contenido = nuevo; }
};

// -------------------
// Clase genérica con puntero
template <typename T>
class CajaPtr {
    T* contenido;  // puntero al objeto
public:
    CajaPtr(T* c) : contenido(c) {}
    T* obtener() { return contenido; }
    void set(T* nuevo) { contenido = nuevo; }
};

// -------------------
// Main de prueba
int main() {
    int x = 10;
    double y = 3.14;

    // Caja por valor
    CajaValor<int> cv(x);
    std::cout << "CajaValor<int>: " << cv.obtener() << "\n";

    // Caja por referencia
    CajaRef<int> cr(x);
    std::cout << "CajaRef<int>: " << cr.obtener() << "\n";
    cr.set(20);
    std::cout << "CajaRef<int> después de set: " << x << "\n";

    // Caja con puntero
    CajaPtr<double> cp(&y);
    std::cout << "CajaPtr<double>: " << *cp.obtener() << "\n";
    double z = 5.5;
    cp.set(&z);
    std::cout << "CajaPtr<double> después de set: " << *cp.obtener() << "\n";
}
