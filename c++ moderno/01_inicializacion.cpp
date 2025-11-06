#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <cstdint>

int main() {
    // inicialización moderna
    int x{5};
    double pi{3.14159};
    bool flag{true};
    char ch{'A'};
    std::string name{"AsdCain"};

    std::cout << "x=" << x << ", pi=" << pi << ", nombre=" << name << "\n";

    // vector con tipo explícito
    std::vector<int> v{1, 2, 3, 4};

    // multiplicar cada elemento por x (usar referencia para modificar)
    for (auto& n : v) {
        n *= x;
    }

    std::cout << "vector multiplicado: ";
    for (const auto& n : v) {
        std::cout << n << ' ';
    }
    std::cout << '\n';

    // std::array: tamaño fijo conocido en tiempo de compilación
    std::array<int, 3> a{10, 20, 30};
    std::cout << "a[1] = " << a[1] << ", size=" << a.size() << '\n';

    // deducción de plantilla (C++17): el compilador infiere vector<double>
    auto ded = std::vector{4.5, 2.0, 3.25};
    std::cout << "deduced vector: ";
    for (auto d : ded) std::cout << d << ' ';
    std::cout << '\n';

    // tamaños con cstdint
    int32_t i32{1000};
    uint8_t u8{255};
    std::cout << "i32=" << i32 << ", u8=" << static_cast<int>(u8) << '\n';

    return 0;
}
