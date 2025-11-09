// Archivo: 09_policy_filter.cpp
#include <iostream>
#include <deque>
#include <cmath>

// Políticas de filtrado
template <typename T>
struct FiltroNulo {
    T aplicar(T v) const noexcept { return v; }
    void actualizar(T) noexcept {}
};

template <typename T, std::size_t N = 3>
struct FiltroMedia {
    std::deque<T> buf;
    T aplicar(T v) {
        buf.push_back(v);
        if (buf.size() > N) buf.pop_front();
        T s = T{};
        for (auto &x : buf) s += x;
        return s / static_cast<T>(buf.size());
    }
    void actualizar(T) noexcept {}
};

template <typename T>
struct FiltroExponencial {
    T alpha;
    bool iniciado = false;
    T estado{};
    FiltroExponencial(T a = T(0.5)) : alpha(a) {}
    T aplicar(T v) {
        if (!iniciado) { estado = v; iniciado = true; return estado; }
        estado = alpha * v + (1 - alpha) * estado;
        return estado;
    }
    void actualizar(T) noexcept { iniciado = false; }
};

// Sensor con política
template <typename T, typename FilterPolicy = FiltroNulo<T>>
class Sensor {
    T valor;
    FilterPolicy filtro;
public:
    Sensor(T v = T{}) : valor(v) {}
    void set(T v) { valor = v; }
    T leer() { return filtro.aplicar(valor); }
    void reset() { filtro.actualizar(valor); }
};

// Uso
int main() {
    Sensor<double, FiltroNulo<double>> s0(10.0);
    std::cout << s0.leer() << "\n";

    Sensor<double, FiltroMedia<double, 3>> s1;
    s1.set(10.0); std::cout << s1.leer() << "\n";
    s1.set(12.0); std::cout << s1.leer() << "\n";
    s1.set(14.0); std::cout << s1.leer() << "\n";

    Sensor<double, FiltroExponencial<double>> s2;
    s2.set(10.0); std::cout << s2.leer() << "\n";
    s2.set(20.0); std::cout << s2.leer() << "\n";
    s2.set(15.0); std::cout << s2.leer() << "\n";

    return 0;
}
