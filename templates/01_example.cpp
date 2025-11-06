/*  
Ejemplo práctico de templates: Sistema de inventario sencillo.

Crea un inventario con productos que tengan cantidad y precio. Implementa funciones template para:

1. Actualizar cantidades:
   - Una versión por referencia.
   - Otra con punteros.

2. Aplicar descuentos o aumentos de precio:
   - Una versión por valor (devuelve nuevo precio).
   - Otra por referencia (modifica precio original).

3. Mostrar información del producto:
   - Sobrecarga de imprimir para valores y punteros.

Conceptos aplicados: templates genéricos, paso por valor, referencia y puntero, sobrecarga de funciones, uso con múltiples tipos (int, double, std::string).
*/

#include <iostream>

template <typename T > 
void print_cambio(T cambio){
    if (cambio > 0){
            std::cout<< "Despues de aumentar: "<< cambio<<std::endl;
        }
    else{
            std::cout<< "Despues de disminuir: "<< cambio<<std::endl;
        }
}

template <typename T > //aunque aqui siempre es entero, template <int N>
void actualizar_cantidad( T &_cantidad, T cambio ){ 
    if ((_cantidad + cambio >= 0) && (cambio != 0)) {
        _cantidad += cambio;
        print_cambio(cambio);
    }
    else{
        std::cout<< "No hay muchos productos o variacion";
   }
}

template <typename T > 
void actualizar_cantidad(T* _cantidad, T cambio){

    if ((*_cantidad + cambio >=0)  && (cambio != 0)) {
        *_cantidad += cambio;
        print_cambio(cambio);
    }
    else{
        std::cout<< "No hay muchos productos o variacion";
   }
}


template <typename U>
U descuento_aumento(){}

template <typename T>
void imprimir(){}

int main() {
    auto producto{"Manzana"};
    int cantidad{10};
    auto precio{2.5};
    int * ptr_cantidad = &cantidad; 

    actualizar_cantidad(ptr_cantidad, 5);   // Funciona
    actualizar_cantidad(cantidad, -3);      // Funciona
    actualizar_cantidad(cantidad, -20);     // Error: no hay suficiente stock

    std::cout << "Cantidad final: " << cantidad << std::endl;



    return 0;
}