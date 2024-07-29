# Video-Analytics

Este proyecto es una aplicación de análisis de video utilizando modelos de detección de objetos basados en YOLO y otros algoritmos.

## Requisitos

- Python 3.8 o superior
- pip (administrador de paquetes de Python)
- Entorno virtual (opcional pero recomendado)

## Instalación

### 1. Clonar el repositorio

git clone <URL_DEL_REPOSITORIO>
cd video-analytics


##  Crear y activar un entorno virtual 
python -m venv env
env\Scripts\activate

## Instalar las dependencias
pip install -r requirements.txt

## configurar YOLO
Descargar pesos y configuraciones de YOLOv4 y YOLOv5
Coloca los archivos yolov4.weights y yolov4.cfg en el directorio yolov5/.
Descarga los pesos necesarios para YOLOv5 y colócalos en el mismo directorio yolov5/.


## Entrenamiento del modelo
python train_model.py



