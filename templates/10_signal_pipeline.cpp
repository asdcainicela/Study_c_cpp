// Archivo: 10_signal_pipeline.cpp
#include <iostream>
#include <vector>
#include <functional>

template <typename T>
class Pipeline {
    std::vector<std::function<T(T)>> stages;
public:
    template <typename F>
    void agregar(F f) { stages.emplace_back(f); }

    T ejecutar(T input) const {
        T v = input;
        for (auto &s : stages) v = s(v);
        return v;
    }
};

// Etapas (functors)
template <typename T>
struct EtapaEscala { T factor; EtapaEscala(T f):factor(f){} T operator()(T in) const { return in * factor; } };

template <typename T>
struct EtapaOffset { T off; EtapaOffset(T o):off(o){} T operator()(T in) const { return in + off; } };

template <typename T>
struct EtapaLimite { T minv, maxv; EtapaLimite(T lo, T hi):minv(lo),maxv(hi){} T operator()(T in) const {
    if (in < minv) return minv;
    if (in > maxv) return maxv;
    return in;
} };

int main() {
    Pipeline<double> p;
    p.agregar(EtapaEscala<double>(2.0));
    p.agregar(EtapaOffset<double>(-1.5));
    p.agregar(EtapaLimite<double>(0.0, 10.0));

    double entrada = 3.0;
    double salida = p.ejecutar(entrada);
    std::cout << salida << "\n";

    // Con lambda
    Pipeline<int> p2;
    p2.agregar([](int x){ return x + 5; });
    p2.agregar([](int x){ return x * 3; });
    std::cout << p2.ejecutar(2) << "\n";
    return 0;
}
