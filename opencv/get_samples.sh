#!/bin/bash
# ===========================================
# Descarga imágenes y videos de muestra OpenCV (FULL)
# Autor: AsdCain
# ===========================================

set -e
echo "  Creando carpetas..."
mkdir -p data videos

# -----------------------------
#   IMÁGENES
# -----------------------------
echo "  Descargando imágenes base..."
declare -A images
images=(
  ["lena.jpg"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
  ["fruits.jpg"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg"
  ["shapes.png"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/smarties.png"
  ["gray.jpg"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg"
  ["building.jpg"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg"
  ["road.png"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/road.png"
  ["coins.png"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/coins.png"
  ["face.jpg"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/face.jpg"
  ["messi.jpg"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg"
)

for name in "${!images[@]}"; do
  url="${images[$name]}"
  echo "-> $name"
  wget -q -c -O "data/$name" "$url" || echo "  Error descargando $name"
done

# -----------------------------
# 🎥 VIDEOS
# -----------------------------
echo "🎥 Descargando videos..."
declare -A videos
videos=(
  ["traffic.avi"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"
  ["walk_big_bunny.mp4"]="https://github.com/opencv/opencv_extra/raw/master/testdata/highgui/video/big_buck_bunny.mp4"
  ["people.avi"]="https://github.com/opencv/opencv_extra/raw/master/testdata/highgui/video/768x576.avi"
  ["cars.mp4"]="https://github.com/opencv/opencv_extra/raw/master/testdata/highgui/video/cars.mp4"
  ["drone.mp4"]="https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.mp4"
  ["pedestrians.mp4"]="https://github.com/intel-iot-devkit/sample-videos/raw/master/pedestrians-detection.mp4"
  ["construction_zone.mp4"]="https://github.com/intel-iot-devkit/sample-videos/raw/master/worker-zone-detection.mp4"
  ["street_scene.mp4"]="https://files.sampleswap.org/SAMPLE-SOUNDS-VIDEO/VIDEO_STREET_SCENE_480P.mp4"
  ["office_ocean.mp4"]="https://filesamples.com/samples/video/mp4/sample_960x400_ocean_with_audio.mp4"
  ["sample_640x360.mp4"]="https://filesamples.com/samples/video/mp4/sample_640x360.mp4"
  ["city_traffic.mp4"]="https://cdn.pixabay.com/vimeo/266805907/traffic-15484.mp4?width=1280&hash=2e3c7a894f3a3fa5d57ed3af2c98f8a8c4abbd2b"
  ["crowd.mp4"]="https://cdn.pixabay.com/vimeo/266805939/crowd-15487.mp4?width=1280&hash=ac4b9622cfd98dcae1b0a47044937e3036da6b4b"
  ["bike_street.mp4"]="https://cdn.pixabay.com/vimeo/217111414/bikes-9129.mp4?width=1280&hash=2cf214e83a904b94f31a5af6d51c8c9acda57bfa"
  ["night_traffic.mp4"]="https://cdn.pixabay.com/vimeo/251805271/night-traffic-13701.mp4?width=1280&hash=16857d44a34b36e64dceef9242492ad8ef166eb4"
  ["drone_highway.mp4"]="https://cdn.pixabay.com/vimeo/210732774/drone-7882.mp4?width=1280&hash=3b894e1fbc91df28b91cfddfa246fc882c61b0e1"
)

for name in "${!videos[@]}"; do
  url="${videos[$name]}"
  echo "-> $name"
  wget -c --show-progress -O "videos/$name" "$url" || echo "⚠️ Error descargando $name"
done
 
echo
echo " Descarga completa!"
echo "-------------------------------------"
echo "  Imágenes: $(ls data | wc -l)"
echo "  Videos:   $(ls videos | wc -l)"
echo "Archivos guardados en:"
echo "  $(pwd)/data"
echo "  $(pwd)/videos"
echo "-------------------------------------"
