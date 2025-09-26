#include <iostream>
int main() {
    int x{42};
    int* p{&x};

    std::cout << "x = " << x << '\n';     // 42
    std::cout << "*p = " << *p << '\n';   // 42
    std::cout << "p = " << p << '\n';     // dirección de x

    *p = 100;   // cambio el valor a través del puntero

    std::cout << "x = " << x << '\n';     // 100
    std::cout << "*p = " << *p << '\n';   // 100
    std::cout << "p = " << p << '\n';     // dirección de x
}
