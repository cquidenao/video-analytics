import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, Response, jsonify
from werkzeug.utils import secure_filename
import csv
import numpy as np
import cv2
import torch
import json
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Crear la carpeta de uploads si no existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Verificar si el archivo de pesos existe
weights_path = 'yolov5/runs/train/exp5/weights/best.pt'  # Actualiza esta línea con la ruta correcta
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"El archivo de pesos no se encuentra en la ruta especificada: {weights_path}")

# Cargar YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

# Cargar las traducciones de etiquetas
labels_es = {
  "0": "amortiguador",
  "1": "reductor",
  "2": "chicote",
  "3": "panel solar",
  "4": "tracker",
  "5": "tornillos",
  "6": "puesta a tierra",
  "7": "rodamientos",
  "8": "ejes",
  "9": "cojinete",
  "10": "bastidor",
  "11": "soporte Simple",
  "12": "cables",
  "13": "conector",
  "14": "actuadores",
  "15": "estructura de soporte",
  "16": "torques",
  "17": "conector suelto",
  "18": "torque alineado",
  "19": "torque desalineado",
  "20": "amortiguador suelto"

}

def save_error_to_csv(error):
    with open('errors.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([error])

def process_frame(frame):
    # Convertir el frame a formato compatible con YOLO
    results = model(frame)
    detections = results.pandas().xyxy[0]
    errors = []
    alerts = []

    for index, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']
        errors.append(f"{label}: {confidence:.2f}")

        # Agregar alerta si se detecta algo en mal estado
        if label in ["conector suelto", "torque desalineado", "amortiguador suelto"]:
            alerts.append(f"ALERTA: {label} detectado con {confidence:.2f} de confianza")

        # Seleccionar el color del cuadro delimitador
        # Etiquetas que deben ser verdes
        etiquetas_verdes = ["torque alineado", "conector", "panel solar", "cojinete", "chicote", "reductor", "amortiguador bueno"]
        color = (0, 255, 0) if label in etiquetas_verdes else (0, 0, 255)  # Verde para alineado/conectado/panel solar, rojo para desalineado/suelto

        # Dibujar la caja delimitadora
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame, errors, alerts

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({"filename": filename})

@app.route('/analyze/<filename>')
def analyze_video_stream(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results/<filename>')
def get_results(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    errors = []
    alerts = []
    cap = cv2.VideoCapture(video_path)

    while True:
        success, frame = cap.read()
        if not success:
            break

        _, frame_errors, frame_alerts = process_frame(frame)
        errors.extend(frame_errors)
        alerts.extend(frame_alerts)

    cap.release()
    return jsonify({"errors": errors, "alerts": alerts})

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    all_errors = []
    all_alerts = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Procesar el frame
        frame, errors, alerts = process_frame(frame)
        all_errors.extend(errors)
        all_alerts.extend(alerts)

        # Codificar el frame en formato JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Ralentizar el procesamiento
        time.sleep(0.1)

    cap.release()
    return all_errors, all_alerts

if __name__ == '__main__':
    app.run(debug=True)
