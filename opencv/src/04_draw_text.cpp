#include <opencv2/opencv.hpp>
#include <iostream>

int main(){

    cv::Mat img(1280, 720, CV_8UC3, cv::Scalar(255, 255, 255)); // Imagen en blanco


    cv::imshow("Canvas", img);
    cv::waitKey(0);
    return 0;

}