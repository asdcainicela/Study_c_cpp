#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {5, 3, 9, 1};

    std::sort(v.begin(), v.end());       // ordena ascendente
    std::reverse(v.begin(), v.end());    // invierte el orden

    auto it = std::find(v.begin(), v.end(), 5);
    if (it != v.end())
        *it = 50;

    for (auto x : v)
        std::cout << x << " ";
    std::cout << std::endl;

    return 0;
}
