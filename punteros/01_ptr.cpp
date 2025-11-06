#include <iostream>
int main() {
    int x{42};
    int* p{&x};

    std::cout << "x = " << x << '\n';     // 42
    std::cout << "Direccion de x (&x): " << &x << '\n';
    std::cout << "*p = " << *p << '\n';   // 42
    std::cout << "p = " << p << '\n';     // dirección de x

}
