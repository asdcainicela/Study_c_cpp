/*
Crea std::vector<int> con valores del 1 al 10 usando inicializador {...}. 
Multiplica cada elemento por 3 usando un for (auto& ...) y muestra el resultado.
Pista: recuerda auto& para modificar en sitio.
*/

#include<iostream>
#include <vector>

int main(){
    int multiplicador{3};

    std::vector<int> valores{1,2,3,4,5,6,7,8,9,10};

    for (auto &n : valores){
        n*=multiplicador;
        std::cout << n << ' ';
    }

    return 0;
}