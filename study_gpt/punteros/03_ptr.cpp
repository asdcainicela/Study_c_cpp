#include <iostream> 

int main() {
    int arr[5] = {10, 20, 30, 40, 50};
    int* p = arr; // arr = &arr[0]

    std::cout << "primer elemento: " << *p << std::endl; // 10
    std::cout << "*p: " << *p << std::endl;   // 10
    std::cout << "segundo elemento: " << *(p + 1) << std::endl; // 20
    for (int i = 0; i < 5; ++i) {
        std::cout << "Elemento " << i << ": " << *(p + i) << std::endl;
    }

    return 0;
}
