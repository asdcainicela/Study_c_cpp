#include <iostream>
#include <chrono>

int main() {
    std::chrono::seconds s(5); // 5 segundos

    std::cout << "s: " << s.count() << " segundos\n";
}
