#include <iostream>
using namespace std;

int main() {
    int* p = new int; // reserva un entero en heap
    *p = 100;

    cout << "Valor en heap: " << *p << endl;

    delete p; // libera memoria
    return 0;
}
