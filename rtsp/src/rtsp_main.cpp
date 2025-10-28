#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>

std::string camera_url(const std::string& user, const std::string& pass, const std::string& ip, int port) {
    return "rtsp://" + user + ":" + pass + "@" + ip + ":" + std::to_string(port) + "/main";
}

cv::VideoCapture open_cap(const std::string& url, int retries=5) {
    cv::VideoCapture cap;
    for (int i = 0; i < retries; ++i) {
        cap.open(url, cv::CAP_FFMPEG);
        if (cap.isOpened()) return cap;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    throw std::runtime_error("No se pudo conectar a: " + url);
}

int main() {
    auto start_main = std::chrono::steady_clock::now();
    std::string url = camera_url("admin", "Panto2025", "192.168.0.101", 554);
    
    cv::VideoCapture cap;
    try { cap = open_cap(url); }
    catch (const std::exception& e) { std::cerr << e.what() << "\n"; return -1; }

    cv::Mat frame, display;
    int frames = 0, lost = 0;
    auto start = std::chrono::steady_clock::now();

    while (true) {
        if (!cap.read(frame)) {
            lost++;
            cap.release();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            cap = open_cap(url);
            continue;
        }
        frames++;

        cv::resize(frame, display, cv::Size(640, 360));

        double fps = frames / std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        cv::putText(display, "Frames: " + std::to_string(frames) + " FPS: " + std::to_string(int(fps)) + " Lost: " + std::to_string(lost),
                    {10,30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0,255,0}, 2);

        cv::imshow("RTSP Stream", display);
        char c = (char)cv::waitKey(1);
        if (c == 27 || c == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();

    auto end_main = std::chrono::steady_clock::now();
    std::cout << "Duración total del main: "
              << std::chrono::duration<double>(end_main - start_main).count() << " s\n";
    std::cout << "Frames totales: " << frames << " | Frames perdidos: " << lost << "\n";

    return 0;
}
