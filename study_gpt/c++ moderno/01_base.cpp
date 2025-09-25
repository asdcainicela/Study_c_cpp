#include<iostream>
#include <vector>

int main() {
    int x{5};
    double y{3.14};
    std::vector<int> v{1, 2, 3, 4};

    for (auto i : v){
        std::cout << i*x<< std::endl;
    }
    
    std::cout<< "pi: "<<y<< std::endl;
    return 0;
}