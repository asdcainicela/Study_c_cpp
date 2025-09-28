#include <iostream> 

int main() {
    int x = 10;
    int* p = &x;

    *p = 20;  // cambia el valor de x a través del puntero

    std::cout << "x: " << x << std::endl;     // 20
    std::cout << "*p: " << *p << std::endl;   // 20

    return 0;
}
