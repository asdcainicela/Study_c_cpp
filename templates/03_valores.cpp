#include <iostream>
#include <vector>
#include <string>

template <typename T>
T promedio(const std::vector<T>& datos) {
    T suma = 0;
    for (const auto& v : datos) suma += v;
    return datos.empty() ? 0 : suma / datos.size();
}

int main() {
    std::vector<float> lecturas = {23.5f, 25.2f, 22.8f, 24.1f};
    std::cout << "Promedio temperatura: " << promedio(lecturas) << "\n";

    std::vector<int> muestras = {100, 120, 80, 90};
    std::cout << "Promedio corriente: " << promedio(muestras) << "\n";
    return 0;
}
