#include <iostream>
#include <vector>
#include <numeric> // std::accumulate
#include <utility> // std::move

int main() {
    std::vector<int> v = {1, 2, 3, 4};

    int sum = std::accumulate(v.begin(), v.end(), 0);
    std::cout << "Suma: " << sum << std::endl;

    v.emplace_back(10);                  // inserta al final construyendo el objeto
    std::cout << "Último: " << v.back() << std::endl;

    v.shrink_to_fit();                   // ajusta la capacidad al tamaño real

    std::vector<int> copia = v;          // copia completa
    std::vector<int> movido = std::move(v); // mueve contenido

    std::cout << "copia size = " << copia.size() << " movido size = " << movido.size() << std::endl;
    std::cout << "v vacío? " << std::boolalpha << v.empty() << std::endl;

    return 0;
}
