#include <iostream>
#include <cuda_runtime.h>

// 1️⃣ Definimos un kernel (función que corre en GPU)
__global__ void addTen(float* data, int n) {
    int idx = threadIdx.x; // ID del hilo actual
    if (idx < n) {
        data[idx] += 10.0f;
    }
}

int main() {
    int N = 5;
    float h_data[5] = {1, 2, 3, 4, 5};
    float* d_data;

    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // 2️⃣ Lanzamos el kernel con 1 bloque y N hilos
    addTen<<<1, N>>>(d_data, N);

    // 3️⃣ Sincronizamos GPU para esperar a que termine
    cudaDeviceSynchronize();

    // 4️⃣ Traemos resultados
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 5️⃣ Mostramos el resultado
    for (int i = 0; i < N; ++i)
        std::cout << h_data[i] << " ";

    cudaFree(d_data);
    return 0;
}
