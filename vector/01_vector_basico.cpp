#include <iostream>
#include <vector>

int main() {
    std::vector<int> v1;                 // vector vacío
    std::vector<int> v2(5, 10);          // 5 elementos con valor 10
    std::vector<int> v3 = {1, 2, 3, 4};  // inicialización directa
    
    std::vector<std::string> names(5, "a");
    
    names[2] = "asd";
    std::cout << "v3[0] = " << v3[0] << std::endl;
    std::cout << "names[2] = " << names[2] << std::endl;
    std::cout << "v3.at(2) = " << v3.at(2) << std::endl;
    std::cout << "size = " << v3.size() << " capacity = " << v3.capacity() << std::endl;
    std::cout << "   =" << names.size() << "            = " << names.capacity() << std::endl;
    std::cout << "empty = " << std::boolalpha << v3.empty() << std::endl;

    return 0;
}
