import torch
import cv2
import numpy as np
from pathlib import Path

# Cargar el modelo entrenado
def cargar_modelo(ruta_modelo):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=ruta_modelo)
        model.eval()  # Poner el modelo en modo evaluación
        print("Modelo cargado correctamente.")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        exit()

# Preprocesar el frame para el modelo
def preprocesar_frame(frame):
    # YOLOv5 maneja el preprocesamiento internamente
    return frame

# Procesar las predicciones del modelo
def procesar_predicciones(predictions, frame, detecciones):
    # YOLOv5 devuelve predicciones en un formato específico
    for pred in predictions.xyxy[0]:  # Asumiendo que solo hay una imagen procesada a la vez
        x1, y1, x2, y2, conf, cls = pred
        if conf > 0.5:  # Umbral de confianza
            label = f"{model.names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Objeto detectado: {model.names[int(cls)]} con confianza: {conf:.2f}")
            detecciones.append({
                'clase': model.names[int(cls)],
                'confianza': float(conf),
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
            })

# Mostrar el frame con las detecciones
def mostrar_frame(frame):
    cv2.imshow('Frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        return False
    return True

# Procesar el video frame por frame
def procesar_video(ruta_video, model):
    cap = cv2.VideoCapture(ruta_video)

    if not cap.isOpened():
        print("Error: No se pudo abrir el archivo de video.")
        return

    detecciones = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            print(f"Fin del video o no se pudo leer el frame en el frame {frame_count}.")
            break

        frame_processed = preprocesar_frame(frame)
        
        try:
            results = model(frame_processed)
            procesar_predicciones(results, frame, detecciones)
        except Exception as e:
            print(f"Error al hacer la predicción en el frame {frame_count}: {e}")
            continue

        if not mostrar_frame(frame):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Procesamiento del video completado.")
    
    # Guardar las detecciones en un archivo
    with open('detecciones.json', 'w') as f:
        import json
        json.dump(detecciones, f, indent=4)
    print("Detecciones guardadas en detecciones.json")

if __name__ == "__main__":
    ruta_modelo = 'yolov5/runs/train/exp14/weights/best.pt'
    ruta_video = 'uploads/Amortiguador_suelto_3.mp4'

    modelo = cargar_modelo(ruta_modelo)
    procesar_video(ruta_video, modelo)
