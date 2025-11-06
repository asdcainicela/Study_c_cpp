#include <iostream>
using namespace std;

void duplicar(int* n) {
    *n = *n * 2;
}

void triplicar (int* n){
    *n =*n*3;
}
void sumar5(int* n){
    *n+=5;
}

bool esprimo(int* n){
    if(*n < 2) {
        return false;
    }
    for (int i = 2; i <= *n / 2; i++) {
        if (*n % i == 0) {
            return false;
        }
    }
    return true;
}

int main() {
    int x = 5;
    duplicar(&x);
    cout << "x duplicado: " << x << endl; // 10
    triplicar(&x);
    cout << "x new triplicado: " << x << endl; // 30
    sumar5(&x);
    cout << "x new sumado 5: " << x << endl; // 35
    if (!esprimo(&x)) {
        cout << x << " no es primo" << endl;
    } else {
        cout << x << " es primo" << endl;
    }
}
