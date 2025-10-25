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

    cv::Mat img2, img3, img4, img5, img6;
    cv::blur(img1, img2,cv::Size(5,5)); //cv::blur(src, dst, cv::Size(5,5)); src -> img original, el otro salida

    cv::Canny(img1, img3, 50, 150); //cv::Canny(src, edges, 50, 150); edge es binaria salida y 50-150 thresholds min max

    cv::cvtColor(img1, img4, cv::COLOR_BGR2GRAY); //cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::threshold(img4, img5, 128, 255, cv::THRESH_BINARY); //cv::threshold(gray, binary, 128, 255, cv::THRESH_BINARY);

    cv::medianBlur(img1, img6, 5); //cv::medianBlur(src, dst, 5);

    cv::Mat img_final(tile_size.height, tile_size.width*6, img1.type(), cv::Scalar(0,0,0));
    img1.copyTo(img_final(cv::Rect(0, 0, tile_size.width, tile_size.height)));
    img2.copyTo(img_final(cv::Rect(tile_size.width, 0, tile_size.width, tile_size.height)));
    img3.copyTo(img_final(cv::Rect(tile_size.width*2, 0, tile_size.width, tile_size.height)));
    img4.copyTo(img_final(cv::Rect(tile_size.width*3, 0, tile_size.width, tile_size.height)));
    img5.copyTo(img_final(cv::Rect(tile_size.width*4, 0, tile_size.width, tile_size.height)));
    img6.copyTo(img_final(cv::Rect(tile_size.width*5, 0, tile_size.width, tile_size.height)));
    
    cv::imshow("imagenes con filtros", img_final);
    cv::waitKey(10000);
    return 0;
}