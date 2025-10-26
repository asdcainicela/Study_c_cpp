#include <opencv2/opencv.hpp>
#include <iostream>

int main(){

    cv::Mat img(720, 1280, CV_8UC3, cv::Scalar(255, 255, 255)); // Imagen en blanco

    //cv::line(image, start_point, end_point, color, thickness);
    cv::line(img, cv::Point(100,100), cv::Point(500,100), cv::Scalar(255,0,0), 5); // Línea azul

    cv::rectangle(img, cv::Point(350,100), cv::Point(550,300), cv::Scalar(0,200,200), cv::FILLED); // Rectángulo amarillo relleno
    cv::rectangle(img, cv::Point(600,100), cv::Point(800,300), cv::Scalar(0,0,255), 3); // Rectángulo rojo no relleno

    cv::circle(img, cv::Point(300,200), 50, cv::Scalar(255,0,0), 3);
    cv::circle(img, cv::Point(700,200), 75, cv::Scalar(0,255,0), cv::FILLED);

    //cv::putText(image, text, org, fontFace, fontScale, color, thickness);
    cv::putText(img, "Hola Capybara", cv::Point(150,380),
            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,0), 2);

    cv::putText(img, "Cooker is life, is love", cv::Point(450,500),
            cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, cv::Scalar(100,0,255), 3);
    /*
    FONT_HERSHEY_SIMPLEX
    FONT_HERSHEY_DUPLEX
    FONT_HERSHEY_COMPLEX
    FONT_HERSHEY_SCRIPT_SIMPLEX
    */

    cv::imshow("Canvas", img);
    cv::waitKey(0);
    return 0;

}