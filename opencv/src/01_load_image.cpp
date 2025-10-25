#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::imread("../data/lena.jpg");

    if (img.empty()) {
        std::cerr << "No se pudo cargar la imagen" << std::endl;
        return -1;
    }

    cv::imshow("Imagen cargada", img);
    cv::waitKey(0);
    return 0;
}
