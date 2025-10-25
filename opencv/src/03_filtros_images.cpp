#include <opencv2/opencv.hpp>
#include <iostream>

int main(){
    cv::Mat img1 = cv::imread("../data/board.jpg");
    cv::Size tile_size(320, 240);
    cv::resize(img1, img1, tile_size);
     if (img1.empty()){
        std::cerr << "Error loading image p bro xd" << std::endl;
        return -1;
    }
    cv::blur(img1, img1,cv::Size(5,5));
    /*
    cv::Mat img2 = img1.clone();

    a) Blur / GaussianBlur

blur() → suaviza la imagen promedio de los píxeles vecinos (promedio simple).

cv::blur(src, dst, cv::Size(5,5));


src → imagen original

dst → imagen de salida

Size(5,5) → tamaño del kernel (vecinos que se promedian)

GaussianBlur() → suaviza con un kernel gaussiano, más natural que blur().

cv::GaussianBlur(src, dst, cv::Size(5,5), 1.5);

b) Canny (detección de bordes)

Detecta los bordes fuertes de una imagen.

cv::Canny(src, edges, 50, 150);


50 y 150 → thresholds mínimo y máximo

La salida edges es binaria, 0 (negro) = sin borde, 255 (blanco) = borde

c) cvtColor (cambiar espacio de color)

Convierte entre espacios de color, por ejemplo BGR → GRAY

cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

d) Threshold / AdaptiveThreshold

Convierte imágenes a blanco y negro según un valor umbral.

cv::threshold(gray, binary, 128, 255, cv::THRESH_BINARY);


128 → valor de umbral

255 → valor que toman los píxeles mayores al umbral

THRESH_BINARY → tipo de binarización*/
    cv::imshow(img1);
    cv::waitKey(0);
    return 0;
}