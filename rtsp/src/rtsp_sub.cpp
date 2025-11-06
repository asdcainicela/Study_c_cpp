#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>

std::string gst_pipeline(const std::string& user, const std::string& pass, const std::string& ip, int port, const std::string& stream = "sub") {
    // Usa stream secundario por defecto (más rápido)
    return "rtspsrc location=rtsp://" + user + ":" + pass + "@" + ip + ":" + std::to_string(port) + 
           "/" + stream + " latency=20 ! "
           "rtph264depay ! h264parse ! nvv4l2decoder ! "
           "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink";
}

cv::VideoCapture open_cap(const std::string& pipeline, int retries=5) {
    cv::VideoCapture cap;
    for (int i = 0; i < retries; ++i) {
        cap.open(pipeline, cv::CAP_GSTREAMER);
        if (cap.isOpened()) {
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            std::cout << "✓ Conectado exitosamente\n";
            return cap;
        }
        std::cerr << "Intento " << (i+1) << "/" << retries << " fallido. Reintentando en 2s...\n";
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    throw std::runtime_error("No se pudo conectar a: " + pipeline);
}

int main() {
    auto start_main = std::chrono::steady_clock::now();

    std::string user = "admin", pass = "Panto2025", ip = "192.168.0.101";
    int port = 554;
    
    // Usa "sub" para stream secundario (21 FPS), o "main" para 4K (3 FPS)
    std::string pipeline = gst_pipeline(user, pass, ip, port, "sub");
    
    std::cout << "Pipeline: " << pipeline << "\n\n";

    cv::VideoCapture cap;
    try { 
        cap = open_cap(pipeline); 
    }
    catch (const std::exception& e) { 
        std::cerr << "✗ Error: " << e.what() << "\n"; 
        return -1; 
    }

    cv::Mat frame;
    int frames = 0, lost = 0;
    auto start_fps = std::chrono::steady_clock::now();

    cv::namedWindow("RTSP Stream", cv::WINDOW_NORMAL);

    while (true) {
        if (!cap.read(frame)) {
            lost++;
            std::cerr << "Frame perdido. Reconectando...\n";
            cap.release();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            try {
                cap = open_cap(pipeline);
            } catch (...) {
                std::cerr << "✗ Reconexión fallida\n";
                break;
            }
            continue;
        }

        frames++;
        cv::imshow("RTSP Stream", frame);
        
        char c = (char)cv::waitKey(1);
        if (c == 27 || c == 'q') break;

        // Estadísticas cada 20 frames (~1 segundo a 20 FPS)
        if (frames % 20 == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_fps).count();
            double fps = 20.0 / elapsed;
            
            std::cout << "Frames: " << frames 
                      << " | FPS: " << int(fps) 
                      << " | Perdidos: " << lost << "\n";
            
            start_fps = now;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    auto end_main = std::chrono::steady_clock::now();
    std::cout << "\n=== Estadísticas finales ===\n";
    std::cout << "Duración total: " << std::chrono::duration<double>(end_main - start_main).count() << " s\n";
    std::cout << "Frames totales: " << frames << " | Frames perdidos: " << lost << "\n";
    std::cout << "FPS promedio: " << frames / std::chrono::duration<double>(end_main - start_main).count() << "\n";

    return 0;
}