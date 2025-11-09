// Archivo: 07_traits_sensor.cpp
#include <iostream>
#include <type_traits>
#include <cmath>

template <typename T>
struct SensorTraits {
    static constexpr const char* tipo = "Desconocido";
    static constexpr bool es_flotante = false;
    static constexpr bool es_entero = false;
};

template <>
struct SensorTraits<int> {
    static constexpr const char* tipo = "Entero";
    static constexpr bool es_entero = true;
};

template <>
struct SensorTraits<float> {
    static constexpr const char* tipo = "Flotante simple";
    static constexpr bool es_flotante = true;
};

template <>
struct SensorTraits<double> {
    static constexpr const char* tipo = "Flotante doble";
    static constexpr bool es_flotante = true;
};

template <typename T>
class Sensor {
    T valor;
public:
    explicit Sensor(T v) : valor(v) {}

    template <typename U = T>
    typename std::enable_if<SensorTraits<U>::es_entero, void>::type
    filtrarRuido() {
        if (valor % 2 != 0) valor++;
    }

    template <typename U = T>
    typename std::enable_if<SensorTraits<U>::es_flotante, void>::type
    filtrarRuido() {
        valor = std::round(valor * 10.0) / 10.0;
    }

    template <typename U = T>
    typename std::enable_if<SensorTraits<U>::es_entero, void>::type
    calibrar(U offset) {
        valor += offset;
    }

    template <typename U = T>
    typename std::enable_if<SensorTraits<U>::es_flotante, void>::type
    calibrar(U factor) {
        valor *= factor;
    }

    T leer() const { return valor; }

    void info() const {
        std::cout << SensorTraits<T>::tipo << " -> " << valor << std::endl;
    }
};

int main() {
    Sensor<int> s1(9);
    Sensor<float> s2(25.67f);
    Sensor<double> s3(1024.432);

    s1.filtrarRuido();
    s2.filtrarRuido();
    s3.filtrarRuido();

    s1.calibrar(5);
    s2.calibrar(0.95f);
    s3.calibrar(1.02);

    s1.info();
    s2.info();
    s3.info();

    static_assert(SensorTraits<int>::es_entero);
    static_assert(SensorTraits<double>::es_flotante);
    return 0;
}
