cmake_minimum_required(VERSION 3.10)
project(JetsonCppDemo)

# C++11 o superior
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Flags de optimización
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")

# Buscar OpenCV
find_package(OpenCV REQUIRED)

# Información
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

add_executable(rtsp_main src/rtsp_main.cpp)
target_link_libraries(rtsp_main ${OpenCV_LIBS} pthread)

# Threads (si es necesario)
find_package(Threads REQUIRED)
target_link_libraries(rtsp_viewer Threads::Threads)