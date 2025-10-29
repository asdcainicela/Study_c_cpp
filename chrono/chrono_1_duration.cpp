#include <iostream>
#include <chrono>

int main() {
    std::chrono::seconds s(5); // 5 segundos
    std::chrono::milliseconds ms(2500); // 2500 milisegundos
    std::chrono::microseconds us(3000000); // 3000000 microsegundos
    std::cout << "s: " << s.count() << " segundos\n";
    std::cout << "ms: " << ms.count() << " milisegundos\n";
    std::cout << "us: " << us.count() << " microsegundos\n";
    
    //std::chrono::duration_cast<TARGET>(DURATION_ORIGINAL)
    auto total = s+std::chrono::duration_cast<std::chrono::seconds>(ms)+std::chrono::duration_cast<std::chrono::seconds>(us);
    std::cout << "Time total: " << total.count() << " segundos\n";

    std::chrono::duration<double> total_sec = s + ms + us;
    std::cout << "Total exacto: " << total_sec.count() << " segundos\n";

}
