/*  
Reto práctico: Registro de sensores con templates

Objetivo: Crear un pequeño sistema de registro de sensores usando templates. 
Cada sensor tiene un nombre, un valor y un estado de alerta.

Clases a crear:

1. SensorValor<T> -> almacena el valor de la lectura del sensor (por valor).
   - Método: actualizarValor(T nuevoValor)
   - Método: aplicarCalibracion(T factor) que multiplica el valor por un factor.

2. SensorRef<T> -> almacena un umbral de alerta (por referencia).
   - Método: modificarUmbral(T nuevoUmbral)

3. SensorPtr<T> -> almacena el nombre del sensor (por puntero).
   - Método: cambiarNombre(T* nuevoNombre)

Funciones:
- Una función template imprimirSensor que reciba los tres tipos de sensores y muestre:
  Nombre: <nombre>, Valor: <valor>, Umbral: <umbral>

Main:
1. Declara variables para nombre del sensor, valor y umbral.
2. Crea los tres sensores usando esas variables.
3. Llama a imprimirSensor para mostrar el estado inicial.
4. Modifica:
   - valor del sensor (actualizar + calibración)
   - umbral (modificar referencia)
   - nombre del sensor (cambiar puntero)
5. Llama a imprimirSensor para mostrar los cambios.

Extras:
- Usar tipos distintos (int, float, double) para valores y umbrales.
- Sobrecargar imprimirSensor para aceptar punteros a los sensores.
- Manejar múltiples sensores usando un vector (opcional).
*/

#include <iostream>
#include <string>

template <typename T>
class SensorValor {
    T valor;
public:
    SensorValor(T v) : valor(v) {}
    void actualizar(T nuevo) { valor = nuevo; }
    void calibrar(T f) { valor *= f; }
    T obtener() const { return valor; }
};

template <typename T>
class SensorRef {
    T& umbral;
public:
    SensorRef(T& u) : umbral(u) {}
    void modificar(T nuevo) { umbral = nuevo; }
    T obtener() const { return umbral; }
};

template <typename T>
class SensorPtr {
    T* nombre;
public:
    SensorPtr(T* n) : nombre(n) {}
    void cambiar(T* n) { nombre = n; }
    T obtener() const { return *nombre; }
};

template <typename T1, typename T2, typename T3>
void imprimir(const SensorPtr<T1>& n, const SensorValor<T2>& v, const SensorRef<T3>& u) {
    std::cout << n.obtener() << " | " << v.obtener() << " | " << u.obtener() << "\n";
}

template <typename T1, typename T2, typename T3>
void imprimir(const SensorPtr<T1>* n, const SensorValor<T2>* v, const SensorRef<T3>* u) {
    std::cout << n->obtener() << " | " << v->obtener() << " | " << u->obtener() << "\n";
}

int main() {
    std::string nombre = "Temperatura";
    float valor = 25.5f;
    double umbral = 30.0;

    SensorPtr<std::string> sNombre(&nombre);
    SensorValor<float> sValor(valor);
    SensorRef<double> sUmbral(umbral);

    imprimir(sNombre, sValor, sUmbral);

    sValor.actualizar(28.0f);
    sValor.calibrar(1.1f);
    sUmbral.modificar(35.0);
    std::string nuevo = "Temp Ambiente";
    sNombre.cambiar(&nuevo);

    imprimir(sNombre, sValor, sUmbral);
    imprimir(&sNombre, &sValor, &sUmbral);

    std::string n2 = "Humedad";
    int v2 = 65, u2 = 80;
    SensorPtr<std::string> sN2(&n2);
    SensorValor<int> sV2(v2);
    SensorRef<int> sU2(u2);
    imprimir(sN2, sV2, sU2);

    std::string n3 = "Presión";
    double v3 = 1013.25;
    float u3 = 1020.0f;
    SensorPtr<std::string> sN3(&n3);
    SensorValor<double> sV3(v3);
    SensorRef<float> sU3(u3);
    imprimir(sN3, sV3, sU3);

    return 0;
}
