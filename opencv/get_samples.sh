#!/bin/bash
# ===========================================
# Descarga imágenes y videos de muestra OpenCV (FULL)
# Autor: AsdCain
# ===========================================

set -e
echo "Creando carpetas..."
mkdir -p data videos

# -----------------------------
# IMAGENES
# -----------------------------
echo "Descargando imagenes base..."
declare -A images
images=(
  ["lena.jpg"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
  ["fruits.jpg"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg"
  ["smarties.png"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/smarties.png"
  ["baboon.jpg"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg"
  ["building.jpg"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg"
  ["messi.jpg"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg"
  ["chessboard.png"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/chessboard.png"
  ["graf1.png"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/graf1.png"
  ["box.png"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/box.png"
  ["rubberwhale1.png"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/rubberwhale1.png"
  ["starry_night.jpg"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/starry_night.jpg"
  ["pic1.png"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/pic1.png"
  ["board.jpg"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/board.jpg"
  ["basketball1.png"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/basketball1.png"
  ["left01.jpg"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/left01.jpg"
  ["right01.jpg"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/right01.jpg"
  ["home.jpg"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/home.jpg"
  ["stuff.jpg"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/stuff.jpg"
)

for name in "${!images[@]}"; do
  url="${images[$name]}"
  echo "  -> $name"
  wget -q -c -O "data/$name" "$url" 2>/dev/null || echo "    [WARN] Error descargando $name"
done

# -----------------------------
# VIDEOS
# -----------------------------
echo ""
echo "Descargando videos..."
declare -A videos
videos=(
  # OpenCV samples (pequeños, siempre funcionan)
  ["traffic.avi"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"
  
  # Sample videos de repositorios publicos en GitHub
  ["big_buck_bunny.mp4"]="https://github.com/mediaelement/mediaelement-files/raw/master/big_buck_bunny.mp4"
  ["sintel_trailer.mp4"]="https://github.com/mediaelement/mediaelement-files/raw/master/sintel-trailer.mp4"
  
  # Videos de test oficiales (ligeros)
  ["sample_640x360.mp4"]="https://sample-videos.com/video123/mp4/240/big_buck_bunny_240p_1mb.mp4"
  ["sample_480p.mp4"]="https://sample-videos.com/video123/mp4/480/big_buck_bunny_480p_1mb.mp4"
  
  # Videos para deteccion de objetos
  ["pedestrian_test.mp4"]="https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"
  ["face_test.mp4"]="https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4"
  ["bottle_detection.mp4"]="https://github.com/intel-iot-devkit/sample-videos/raw/master/bottle-detection.mp4"
  
  # Test patterns
  ["test_pattern.mp4"]="https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
  
  # Videos adicionales de archivo libre
  ["elephants_dream.mp4"]="https://archive.org/download/ElephantsDream/ed_hd.mp4"
)

for name in "${!videos[@]}"; do
  url="${videos[$name]}"
  echo "  -> $name"
  wget --timeout=60 --tries=2 -c -q --show-progress --user-agent="Mozilla/5.0" -O "videos/$name" "$url" 2>/dev/null || {
    echo "    [WARN] Error descargando $name"
    rm -f "videos/$name" 2>/dev/null
  }
done

# -----------------------------
# RESUMEN
# -----------------------------
echo ""
echo "Descarga completa!"
echo "-------------------------------------"
echo "  Imagenes: $(ls data 2>/dev/null | wc -l) archivos"
echo "  Videos:   $(ls videos 2>/dev/null | wc -l) archivos"
echo ""
echo "Ubicacion de archivos:"
echo "  $(pwd)/data"
echo "  $(pwd)/videos"
echo "-------------------------------------"

if command -v du &> /dev/null; then
  echo ""
  echo "Espacio utilizado:"
  du -sh data 2>/dev/null || echo "  data: 0 MB"
  du -sh videos 2>/dev/null || echo "  videos: 0 MB"
fi

echo ""
echo "Listo para usar con OpenCV!"