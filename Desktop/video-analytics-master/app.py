import os
from flask import Flask, request, render_template, Response, redirect, url_for, jsonify, send_file
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import gridfs
import pymongo
import cv2
import torch
import json
from bson import ObjectId
import time
from fpdf import FPDF

# Crear una instancia de la aplicación Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Crear el directorio si no existe
socketio = SocketIO(app)

# Configuración de MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["video_analytics"]
fs = gridfs.GridFS(db)

# Verificar si el archivo de pesos existe
weights_path = 'yolov5/runs/train/exp11/weights/best.pt'
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"El archivo de pesos no se encuentra en la ruta especificada: {weights_path}")

# Cargar YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

# Cargar las traducciones de etiquetas
labels_es = {
    "0": "amortiguador bueno",
    "1": "amortiguador suelto",
    "2": "chicote",
    "3": "chicote suelto",
    "4": "cojinete",
    "5": "cojinete corrido",
    "6": "conector",
    "7": "conector suelto",
    "8": "panel solar",
    "9": "reductor",
    "10": "torque alineado",
    "11": "torque desalineado"
}

def process_frame(frame, current_time):
    results = model(frame)
    detections = results.pandas().xyxy[0]
    errors = []
    alerts = []

    for index, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']
        
        # Color del cuadro según la etiqueta
        if confidence > 0.70:
            if label in ["conector suelto", "torque desalineado", "amortiguador suelto"]:
                errors.append((label, confidence, current_time))
                alerts.append(f"ALERTA: {label} detectado con {confidence:.2f} de confianza")
                color = (0, 0, 255)  # Rojo para errores
            else:
                color = (0, 255, 0)  # Verde para objetos buenos

            class_name = label
            if class_name not in detected_objects:
                detected_objects[class_name] = True
                stats['detections_by_class'][class_name] = stats['detections_by_class'].get(class_name, 0) + 1

        # Mostrar todos los objetos, independientemente de la confianza
        color_display = (0, 255, 0) if label in ["torque alineado", "conector", "panel solar", "cojinete", "chicote", "reductor", "amortiguador bueno"] else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_display, 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_display, 2)

    return frame, detections, errors, alerts

detected_objects = {}
stats = {'detections_by_class': {}, 'errors': {}}
error_times = []

def generate_frames(file_id):
    try:
        file = fs.get(ObjectId(file_id))
    except gridfs.errors.NoFile:
        print(f"No se encontró el archivo con ObjectId: {file_id}")
        return "No se encontró el archivo", 404

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    with open(file_path, 'wb') as f:
        f.write(file.read())

    cap = cv2.VideoCapture(file_path)
    frame_count = 0
    update_interval = 50  # Actualiza las estadísticas cada 50 frames
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Establecer un valor predeterminado si FPS no está disponible
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time = frame_count / fps  # Calcular el tiempo actual en segundos
        frame, detections, errors, alerts = process_frame(frame, current_time)
        error_times.extend(errors)  # Agregar los errores y sus tiempos a la lista

        for error in errors:
            class_name = error[0]
            if class_name in stats['errors']:
                stats['errors'][class_name] += 1
            else:
                stats['errors'][class_name] = 1

        frame_count += 1
        if frame_count % update_interval == 0:
            try:
                db['video_stats'].update_one(
                    {'file_id': ObjectId(file_id)},
                    {'$set': {'stats': stats, 'error_times': error_times}},
                    upsert=True
                )
                socketio.emit('update_stats', {'file_id': str(file_id), 'stats': stats})
                print(f"Estadísticas actualizadas en la base de datos para file_id: {file_id}")
            except Exception as e:
                print(f"Error al actualizar estadísticas en la base de datos: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        time.sleep(0.1)  # Pausa de 100 ms entre frames para ralentizar la reproducción

    cap.release()
    try:
        db['video_stats'].update_one(
            {'file_id': ObjectId(file_id)},
            {'$set': {'stats': stats, 'error_times': error_times}},
            upsert=True
        )
        socketio.emit('update_stats', {'file_id': str(file_id), 'stats': stats})
        print(f"Estadísticas finalizadas en la base de datos para file_id: {file_id}")
    except Exception as e:
        print(f"Error al finalizar estadísticas en la base de datos: {e}")

@app.route('/')
def index():
    return render_template('analyze.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_id = fs.put(file, filename=filename)
        return redirect(url_for('analyze_video', file_id=file_id))

@app.route('/analyze/<file_id>')
def analyze_video(file_id):
    return render_template('analyze.html', file_id=file_id)

@app.route('/video_feed/<file_id>')
def video_feed(file_id):
    return Response(generate_frames(file_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats/<file_id>')
def get_stats(file_id):
    stats = db['video_stats'].find_one({'file_id': ObjectId(file_id)})
    if not stats:
        return jsonify(error="No se encontraron estadísticas para este video.")
    print(f"Estadísticas encontradas para file_id: {file_id}")
    return jsonify(stats['stats'])

@app.route('/download_report/<file_id>')
def download_report(file_id):
    stats = db['video_stats'].find_one({'file_id': ObjectId(file_id)})
    if not stats:
        return "No se encontraron estadísticas para este video.", 404

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Título
    pdf.cell(200, 10, txt="Informe de Detección de Objetos", ln=True, align='C')
    pdf.cell(200, 10, txt="INCLUSIVE GROUP", ln=True, align='C')
    pdf.cell(200, 10, txt="---------------------------", ln=True, align='C')
    pdf.cell(200, 10, txt="", ln=True, align='C')
    
    # Estadísticas
    pdf.cell(200, 10, txt="Estadísticas Generales", ln=True)
    pdf.cell(200, 10, txt="---------------------------", ln=True)
    for key, value in stats['stats']['detections_by_class'].items():
        pdf.cell(200, 10, txt=f"{key}: {value} detecciones", ln=True)
    pdf.cell(200, 10, txt="", ln=True)
    
    # Errores
    pdf.cell(200, 10, txt="Errores Detectados", ln=True)
    pdf.cell(200, 10, txt="---------------------------", ln=True)
    for key, value in stats['stats']['errors'].items():
        pdf.cell(200, 10, txt=f"{key}: {value} errores", ln=True)
    pdf.cell(200, 10, txt="", ln=True)
    
    # Errores con tiempo
    pdf.cell(200, 10, txt="Detalles de Errores", ln=True)
    pdf.cell(200, 10, txt="---------------------------", ln=True)
    for error in stats.get('error_times', []):
        label, confidence, error_time = error
        pdf.cell(200, 10, txt=f"{label} con {confidence:.2f} de confianza en el segundo {error_time:.2f}", ln=True)

    report_path = os.path.join(app.config['UPLOAD_FOLDER'], f"reporte_{file_id}.pdf")
    pdf.output(report_path)

    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
