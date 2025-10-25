cmake_minimum_required(VERSION 3.10)
project(OpenCV_Study)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(load_image src/01_load_image.cpp)
target_link_libraries(load_image ${OpenCV_LIBS})

add_executable(canvas_image src/02_canvas_images.cpp)
target_link_libraries(canvas_image ${OpenCV_LIBS})

add_executable(filtros_image src/03_filtros_images.cpp)
target_link_libraries(filtros_image ${OpenCV_LIBS})