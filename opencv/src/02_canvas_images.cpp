#include <iostream>
#include <opencv2/opencv.hpp>

int main(){
    cv::Mat vect;
    cv::Mat img1 = cv::imread("../data/home.jpg");
    cv::Mat img2 = cv::imread("../data/board.jpg");
    cv::Mat img3 = cv::imread("../data/left01.png");
    cv::Mat img4 = cv::imread("../data/starry_night.jpg");

    if (img1.empty() || img2.empty() || img3.empty() || img4.empty()){
        std::cerr << "Error loading images p bro xd" << std::endl;
        return -1;
    }
     // Redimensionamos para que todas tengan el mismo tamaño
    cv::Size tile_size(320, 240);
    cv::resize(img1, img1, tile_size);
    cv::resize(img2, img2, tile_size);
    cv::resize(img3, img3, tile_size);
    cv::resize(img4, img4, tile_size);


     // Creamos una imagen vacía para el mosaico (2x2)
    int rows = 2, cols = 2;
    cv::Mat canvas(tile_size.height * rows, tile_size.width * cols, img1.type(), cv::Scalar(0,0,0));

    // Pegamos cada imagen en su posición
    img1.copyTo(canvas(Rect(0, 0, tile_size.width, tile_size.height)));
    img2.copyTo(canvas(Rect(tile_size.width, 0, tile_size.width, tile_size.height)));
    img3.copyTo(canvas(Rect(0, tile_size.height, tile_size.width, tile_size.height)));
    img4.copyTo(canvas(Rect(tile_size.width, tile_size.height, tile_size.width, tile_size.height)));

    // Mostramos una sola ventana
    cv::imshow("Mosaico", canvas);
    cv::waitKey(0);

    return 0;
}