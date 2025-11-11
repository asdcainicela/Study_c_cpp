#include <iostream>
#include <vector>
#include <numeric>
#include <utility>

void print_vector(const std::vector<int>& v, const std::string& name) {
    std::cout << name << " [size=" << v.size() << "]: ";
    for (auto t : v) std::cout << t << " ";
    std::cout << "\n";
}

int main() {
    std::vector<int> v = {1, 2, 3, 4};
    print_vector(v, "v inicial");

    int sum = std::accumulate(v.begin(), v.end(), 0); // 0 es el valor inicial de la suma
    std::cout << "Suma: " << sum << "\n";

    v.emplace_back(10);
    print_vector(v, "v tras emplace_back");

    v.shrink_to_fit();
    std::vector<int> copia = v;          
    std::vector<int> movido = std::move(v); 

    print_vector(copia, "copia");
    print_vector(movido, "movido");
    print_vector(v, "v (tras move)");

    std::cout << "v vacio? " << std::boolalpha << v.empty() << "\n";
    return 0;
}
