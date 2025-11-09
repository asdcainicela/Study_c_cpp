// Archivo: 08_sensor_array.cpp
#include <iostream>
#include <vector>
#include <type_traits>
#include <numeric>

template <typename T>
class SensorArray {
    static_assert(std::is_arithmetic<T>::value, "SensorArray requiere tipo numérico.");
    std::vector<T> data;
public:
    void push(T v) { data.push_back(v); }
    std::size_t size() const { return data.size(); }

    T promedio() const {
        if (data.empty()) return T{};
        if (std::is_integral<T>::value) {
            long long s = 0;
            for (typename std::vector<T>::const_reference v : data)
                s += static_cast<long long>(v);
            return static_cast<T>(s / static_cast<long long>(data.size()));
        } else {
            double s = std::accumulate(data.begin(), data.end(), 0.0);
            return static_cast<T>(s / data.size());
        }
    }

    template <typename F>
    void calibrarTodos(F factor) {
        if (std::is_integral<T>::value) {
            for (typename std::vector<T>::reference v : data)
                v = static_cast<T>(v + static_cast<T>(factor));
        } else {
            for (typename std::vector<T>::reference v : data)
                v = static_cast<T>(v * factor);
        }
    }

    T operator[](std::size_t i) const { return data[i]; }
};

int main() {
    SensorArray<int> si;
    si.push(10); si.push(13); si.push(12);
    std::cout << "Promedio int: " << si.promedio() << "\n";
    si.calibrarTodos(2);
    std::cout << "Calibrados int: " << si[0] << " " << si[1] << " " << si[2] << "\n";

    SensorArray<double> sd;
    sd.push(1.5); sd.push(2.5); sd.push(3.0);
    std::cout << "Promedio double: " << sd.promedio() << "\n";
    sd.calibrarTodos(1.1);
    std::cout << "Calibrados double: " << sd[0] << " " << sd[1] << " " << sd[2] << "\n";

    return 0;
}
