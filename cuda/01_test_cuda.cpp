#include <iostream>
#include <cuda_runtime.h>

int main() {
    int N = 5;
    float h_data[5] = {1, 2, 3, 4, 5};  // memoria en CPU
    float* d_data;                      // puntero GPU

    // 1️⃣ Reservar memoria en GPU
    cudaMalloc(&d_data, N * sizeof(float));

    // 2️⃣ Copiar CPU → GPU
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // 3️⃣ Copiar GPU → CPU (para probar)
    float h_result[5];
    cudaMemcpy(h_result, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 4️⃣ Mostrar resultado
    for (int i = 0; i < N; ++i)
        std::cout << h_result[i] << " ";

    // 5️⃣ Liberar memoria
    cudaFree(d_data);

    return 0;
}
