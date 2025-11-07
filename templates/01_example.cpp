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
   - Sobrecarga de imprimir por punteros.

Conceptos aplicados: templates genéricos, paso por valor, referencia y puntero, sobrecarga de funciones, uso con múltiples tipos (int, double, std::string).
*/

#include <iostream>

template <typename T > 
void print_cambio(T cambio,T cantidad_actual){
    if (cambio > 0){
            std::cout<< "Despues de aumentar "<< cambio <<": "<< cantidad_actual <<std::endl;
        }
    else{
            std::cout<< "Despues de disminuir "<< cambio <<": "<< cantidad_actual <<std::endl;
        }
}

template <typename T > //aunque aqui siempre es entero, template <int N>
void actualizar_cantidad( T &_cantidad, T cambio ){ 
    if ((_cantidad + cambio >= 0) && (cambio != 0)) {
        _cantidad += cambio;
        print_cambio(cambio, _cantidad);
    }
    else{
        std::cout<< "No hay muchos productos o variacion";
   }
}

template <typename T > 
void actualizar_cantidad(T* _cantidad, T cambio){

    if ((*_cantidad + cambio >=0)  && (cambio != 0)) {
        *_cantidad += cambio;
        print_cambio(cambio, *_cantidad);
    }
    else{
        std::cout<< "No hay stock" << std::endl;
   }
}

template <typename U>
U descuento_aumento( U precio, U porcentaje){

    if (porcentaje >= 0 ){
        std::cout << "Aplicando aumento ... puede ser " <<  std::endl;

    }
    else if (porcentaje > -100.0){
        std::cout<< "Aplicando descuento ..." << std::endl;   
    }
    else{
        std::cerr<< "No valido" << std::endl;
        return precio;
    }

    U precio_variado =  precio * (1 + 1.0*porcentaje/100);
    std::cout<<"el valor puede ser... " << precio_variado <<std::endl;

    return precio_variado;
}

template <typename U>
void descuento_aumento( U* precio, U porcentaje){
     if (porcentaje >= 0 ){
        std::cout << "Aplicando aumento" << std::endl;
    }
    else if (porcentaje > -100.0){
        std::cout<< "Aplicando descuento ..." << std::endl;   
    }
    else{
        std::cerr<< "No valido" << std::endl;
        return;
    }
    *precio *= 1 + porcentaje * 1.0 / 100;
}

template <typename S, typename T, typename R>
void imprimir(S producto, T* cantidad, R* precio ){
    std::cout<< "tenemos " << *cantidad << " " << producto << " a " << *precio << "soles"<< std::endl;
}

int main() {
    std::string producto{"Manzana"};
    int cantidad{10};
    float precio{2.5};
    int* ptr_cantidad = &cantidad; 
    float* prt_precio = &precio;

    actualizar_cantidad(ptr_cantidad, 5);
    actualizar_cantidad(cantidad, -3);
    //actualizar_cantidad(cantidad, -20);     // no hay suficiente stock
    
    float posible_valor;
    posible_valor =  descuento_aumento<float>(precio, -20.0);
    descuento_aumento<float>(prt_precio, 25.0);

    imprimir<std::string, int, float>(producto, ptr_cantidad, prt_precio);

    return 0;
}