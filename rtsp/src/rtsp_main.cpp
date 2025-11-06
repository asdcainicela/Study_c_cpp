#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>

std::string gst_pipeline(const std::string& user, const std::string& pass, const std::string& ip, int port) {
    return "rtspsrc location=rtsp://" + user + ":" + pass + "@" + ip + ":" + std::to_string(port) + 
           "/main latency=0 ! "  // ← Cambiado de 100 a 0 (baja latencia)
           "rtph264depay ! h264parse ! nvv4l2decoder ! "
           "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
           "appsink drop=true sync=false max-buffers=1";  // ← Optimizado: descarta frames viejos
}

cv::VideoCapture open_cap(const std::string& pipeline, int retries=5) {
    cv::VideoCapture cap;
    for (int i = 0; i < retries; ++i) {
        cap.open(pipeline, cv::CAP_GSTREAMER);
        if (cap.isOpened()) {
            // Reducir buffer interno de OpenCV
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            return cap;
        }
        std::cerr << "Intento " << (i+1) << "/" << retries << " fallido. Reintentando en 2s...\n";
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    throw std::runtime_error("No se pudo conectar a: " + pipeline);
}

int main() {
    auto start_main = std::chrono::steady_clock::now();

    // Suprimir warnings de OpenCV
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    std::string user = "admin", pass = "Panto2025", ip = "192.168.0.101";
    int port = 554;
    std::string pipeline = gst_pipeline(user, pass, ip, port);

    cv::VideoCapture cap;
    try { 
        cap = open_cap(pipeline); 
        std::cout << "✓ Conectado a la cámara\n";
    }
    catch (const std::exception& e) { 
        std::cerr << "✗ Error: " << e.what() << "\n"; 
        return -1; 
    }

    cv::Mat frame, display;
    int frames = 0, lost = 0;
    auto start_fps = std::chrono::steady_clock::now();
    auto last_display = std::chrono::steady_clock::now();

    // Intentar crear ventana (si falla, seguirá sin GUI)
    bool gui_available = true;
    try {
        cv::namedWindow("RTSP Stream", cv::WINDOW_NORMAL);
    } catch (...) {
        gui_available = false;
        std::cout << "⚠ GUI no disponible, corriendo en modo headless\n";
    }

    while (true) {
        // Capturar frame
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

        // Mostrar GUI solo si está disponible y no muy frecuente
        auto now = std::chrono::steady_clock::now();
        if (gui_available && std::chrono::duration<double>(now - last_display).count() > 0.033) {
            // Limitar GUI a ~30 FPS máximo para no bloquear
            cv::resize(frame, display, cv::Size(640, 360));
            
            try {
                cv::imshow("RTSP Stream", display);
                char c = (char)cv::waitKey(1);
                if (c == 27 || c == 'q') break;
            } catch (...) {
                gui_available = false;
                std::cout << "⚠ GUI falló, continuando sin visualización\n";
            }
            
            last_display = now;
        }

        // Estadísticas cada 30 frames
        if (frames % 30 == 0) {
            double fps = 30.0 / std::chrono::duration<double>(now - start_fps).count();
            start_fps = now;
            std::cout << "Frames: " << frames 
                      << " | FPS: " << int(fps) 
                      << " | Perdidos: " << lost 
                      << (gui_available ? " | GUI: ON" : " | GUI: OFF") << "\n";
        }
    }

    cap.release();
    if (gui_available) {
        cv::destroyAllWindows();
    }

    auto end_main = std::chrono::steady_clock::now();
    std::cout << "\n=== Estadísticas finales ===\n";
    std::cout << "Duración total: " << std::chrono::duration<double>(end_main - start_main).count() << " s\n";
    std::cout << "Frames totales: " << frames << " | Frames perdidos: " << lost << "\n";
    std::cout << "FPS promedio: " << frames / std::chrono::duration<double>(end_main - start_main).count() << "\n";

    return 0;
}