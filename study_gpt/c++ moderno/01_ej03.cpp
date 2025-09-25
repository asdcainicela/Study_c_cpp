/*
Partiendo del vector del problema 2, escribe dos bucles:

Uno con for (auto n : v) que intente poner n = 0; y luego imprime v.

Otro con for (auto& n : v) y haz n = 0; e imprime v.
Explica la diferencia en comentarios.
*/

#include<iostream>
#include <vector>

int main(){

    std::vector<int> valores{1,2,3,4,5,6,7,8,9,10};

    for (auto n : valores) {
        n = 0;
        std::cout << n << ' ';
    }
    std::cout << '\n';

    for (auto& n : valores) {
        n = 0;
        std::cout << n << ' ';
    }

    return 0;
}