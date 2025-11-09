#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3};

    v.push_back(4);                // agrega al final
    v.insert(v.begin() + 1, 10);   // inserta en posición 1
    v.pop_back();                  // elimina el último
    v.erase(v.begin());            // elimina el primero

    v.reserve(10);                 // reserva memoria
    v.resize(5, 0);                // cambia tamaño lógico

    std::cout << "size = " << v.size() << " capacity = " << v.capacity() << std::endl;

    for (int n : v)
        std::cout << n << " ";
    std::cout << std::endl;

    return 0;
}
