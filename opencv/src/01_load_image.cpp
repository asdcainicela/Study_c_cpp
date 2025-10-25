#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img_lena = cv::imread("../data/lena.jpg");
    cv::Mat img_fruits = cv::imread("../data/fruits.jpg");

    if (img_lena.empty()) {
        std::cerr << "No se pudo cargar la imagen" << std::endl;
        return -1;
    }

    if (img_fruits.empty()) {
        std::cerr << "Fallo cargar p mano" << std::endl;
        return -1;
    }

    cv::Mat img_fruits_clone = img_fruits.clone(); //clonamos la imagen de fruita
    cv::resize(img_fruits_clone, img_fruits_clone, cv::Size(640,400)); // redimensionamos la imagen clonada

    cv::Mat img_fruits_clone2 = img_fruits_clone.clone(); //clonamos otra vez
    cv::resize(img_fruits_clone2, img_fruits_clone2, cv::Size(500,640)); // redimensionamos la imagen clonada 2

    cv::imshow("Imagen cargada", img_lena);
    cv::moveWindow("Imagen cargada", 0, 0);

    cv::imshow("Imagen fruits load p", img_fruits);
    cv::moveWindow("Imagen fruits load p", 650, 0);

    cv::imshow("Imagen de fruitas clonada", img_fruits_clone);
    cv::moveWindow("Imagen de fruitas clonada", 0, 450);

    cv::imshow("Imagen de fruitas clonada 2", img_fruits_clone2);
    cv::moveWindow("Imagen de fruitas clonada 2", 650, 450);

    cv::waitKey(5000); //podemmos hacer cv::waitKey(0);
    return 0;
}
