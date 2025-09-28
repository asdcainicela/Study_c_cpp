#include <iostream> 

int main() {
    int x = 7;        // variable normal
    int* p = &x;      // puntero a x
    int** pp = &p;    // puntero a puntero (apunta a p)

    // --- Valores ---
    std::cout << "------------- Valores -------------" << std::endl;
    std::cout << "x      : " << x << std::endl;       // valor directo
    std::cout << "*p     : " << *p << std::endl;      // valor usando p
    std::cout << "**pp   : " << **pp << std::endl;    // valor usando pp

    // --- Direcciones ---
    std::cout << "------------- Direcciones -------------" << std::endl;
    std::cout << "&x     : " << &x << std::endl;      // dirección de x
    std::cout << "p      : " << p << std::endl;       // p guarda &x
    std::cout << "*pp    : " << *pp << std::endl;     // *pp = p = &x

    std::cout << "&p     : " << &p << std::endl;      // dirección de p
    std::cout << "pp     : " << pp << std::endl;      // pp guarda &p

    std::cout << "&pp    : " << &pp << std::endl;     // dirección de pp

    return 0;
}
