/*
Declara variables: int, double, bool, char, std::string usando {}. Imprímelas con std::cout.
Pista: usa std::string name{"tuNombre"};.
*/
#include<iostream>

int main(){
    int gpio{5};
    double pi{3.1415};
    bool flag{false};
    char letter{'C'};
    std::string name{"asdCain"};

    std::cout << "gpio=" << gpio << ", pi=" << pi << ", flag=" << flag
              << ", letter=" << letter << ", name=" << name << '\n';

    return 0;
}