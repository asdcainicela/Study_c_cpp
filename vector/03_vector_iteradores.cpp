#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {5, 3, 9, 1};
    std::sort(v.begin(), v.end());       // ordena ascendente
    
    std::cout << "ordenamos: ";
    for (auto n : v){
        std::cout << n << " ";
    }
    std::cout << "\nrevertimos el orden: ";
    
    std::reverse(v.begin(), v.end());    // invierte el orden
    
    for (auto m : v){
        std::cout << m << " ";
    }
    
    std::cout << std::endl;
    
    auto it = std::find(v.begin(), v.end(), 5);
    
    if (it != v.end()) {
        std::cout << "Encontrado en posición: " << (it - v.begin()) << std::endl;
        std::cout << "Valor encontrado: " << *it << std::endl;
        std::cout << "Dirección de memoria: " << &(*it) << std::endl;
        *it = 50;
    } else {
        std::cout << "No encontrado" << std::endl;
    }
    
    std::cout << "Vector después del cambio: ";
    for (auto x : v)
        std::cout << x << " ";
    std::cout << std::endl;
    
    return 0;
}