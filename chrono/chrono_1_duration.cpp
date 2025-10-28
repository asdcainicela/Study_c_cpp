#include <iostream>
#include <chrono>

int main() {
    chrono::seconds s(5); // 5 segundos

    std::cout << "s: "<<s.count() << "segundos\n";


}