#include <iostream>

int main() {
    auto x{5};
    auto& y{x}; // y is a reference to x
    auto* z{&x}; // z is a pointer to an int
    std::cout << "x: " << x << ", y: " << y << ", z: " << *z << std::endl;
    return 0;
}