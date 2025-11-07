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
