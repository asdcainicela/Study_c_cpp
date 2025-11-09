// Archivo: 08_sensor_array.cpp
#include <iostream>
#include <vector>
#include <concepts>
#include <numeric>
#include <cmath>

template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

template <Numeric T>
class SensorArray {
    std::vector<T> data;
public:
    void push(T v) { data.push_back(v); }
    std::size_t size() const { return data.size(); }

    T promedio() const {
        if (data.empty()) return T{};
        if constexpr (std::is_integral_v<T>) {
            long long s = 0;
            for (auto &v : data) s += static_cast<long long>(v);
            return static_cast<T>(s / static_cast<long long>(data.size()));
        } else {
            double s = std::accumulate(data.begin(), data.end(), 0.0);
            return static_cast<T>(s / data.size());
        }
    }

    template <typename F>
    void calibrarTodos(F factor) {
        if constexpr (std::is_integral_v<T>)
            for (auto &v : data) v = static_cast<T>(v + static_cast<T>(factor));
        else
            for (auto &v : data) v = static_cast<T>(v * factor);
    }

    T operator[](std::size_t i) const { return data[i]; }
};

int main() {
    SensorArray<int> si;
    si.push(10); si.push(13); si.push(12);
    std::cout << si.promedio() << "\n";
    si.calibrarTodos(2); // suma 2 para ints
    std::cout << si[0] << " " << si[1] << " " << si[2] << "\n";

    SensorArray<double> sd;
    sd.push(1.5); sd.push(2.5); sd.push(3.0);
    std::cout << sd.promedio() << "\n";
    sd.calibrarTodos(1.1); // multiplica por 1.1 para floating
    std::cout << sd[0] << " " << sd[1] << " " << sd[2] << "\n";
    return 0;
}
