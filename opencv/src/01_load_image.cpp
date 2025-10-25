#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::imread("../data/lena.jpg");
    cv::Mat img2 = cv::imread("../data/fruits.jpg");

    if (img.empty()) {
        std::cerr << "No se pudo cargar la imagen" << std::endl;
        return -1;
    }

    if (img2.empty()){
        std::cerr <<"fallo cargar p mano"<< std::endl;
        return -1;
    }

    cv::imshow("Imagen cargada", img);
    cv::imshow("Imagen fruits load p", img2);
    cv::waitKey(0);
    return 0;
}
