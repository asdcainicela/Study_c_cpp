#!/bin/bash

set -e
echo "Creando carpetas..."
mkdir -p data videos

# -----------------------------
# IMAGENES
# -----------------------------
echo "Descargando imágenes base..."
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
  ["cameraman.jpg"]="https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/cv/shared/lena.png"
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
  # Videos de OpenCV (pequeños y confiables)
  ["traffic.avi"]="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"
  
  # Videos de muestra públicos (Pexels - gratuitos y confiables)
  ["people_walking.mp4"]="https://videos.pexels.com/video-files/3255275/3255275-uhd_2560_1440_25fps.mp4"
  ["city_traffic.mp4"]="https://videos.pexels.com/video-files/2103099/2103099-uhd_2560_1440_24fps.mp4"
  ["crowd_people.mp4"]="https://videos.pexels.com/video-files/6894055/6894055-uhd_2560_1440_25fps.mp4"
  ["cars_highway.mp4"]="https://videos.pexels.com/video-files/2103101/2103101-uhd_2560_1440_24fps.mp4"
  ["pedestrians.mp4"]="https://videos.pexels.com/video-files/8488625/8488625-uhd_2560_1440_25fps.mp4"
  ["street_scene.mp4"]="https://videos.pexels.com/video-files/3843211/3843211-uhd_2560_1440_30fps.mp4"
  ["bike_street.mp4"]="https://videos.pexels.com/video-files/4887366/4887366-uhd_2560_1440_24fps.mp4"
  ["night_city.mp4"]="https://videos.pexels.com/video-files/3129957/3129957-uhd_2560_1440_30fps.mp4"
  ["drone_view.mp4"]="https://videos.pexels.com/video-files/2611250/2611250-uhd_2560_1440_25fps.mp4"
  ["construction.mp4"]="https://videos.pexels.com/video-files/3044127/3044127-uhd_2560_1440_25fps.mp4"
  
  # Videos de ejemplo de Pixabay (gratuitos)
  ["ocean_waves.mp4"]="https://cdn.pixabay.com/video/2016/06/14/3619-170361326_large.mp4"
  ["forest_path.mp4"]="https://cdn.pixabay.com/video/2019/04/23/23139-333782893_large.mp4"
  ["urban_street.mp4"]="https://cdn.pixabay.com/video/2020/03/27/35179-402259001_large.mp4"
  ["sports_action.mp4"]="https://cdn.pixabay.com/video/2016/11/11/6074-191066614_large.mp4"
)

for name in "${!videos[@]}"; do
  url="${videos[$name]}"
  echo "  -> $name"
  # Usar wget con timeout y retry para videos grandes
  wget --timeout=30 --tries=3 -c -q --show-progress -O "videos/$name" "$url" 2>/dev/null || {
    echo "    [WARN] Error descargando $name (puede ser muy grande o no disponible)"
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

# Mostrar tamaño total
if command -v du &> /dev/null; then
  echo ""
  echo "Espacio utilizado:"
  du -sh data 2>/dev/null || echo "  data: 0 MB"
  du -sh videos 2>/dev/null || echo "  videos: 0 MB"
fi

echo ""
echo "Listo para usar con OpenCV!"