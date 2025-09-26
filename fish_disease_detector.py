import os
from PIL import Image, ImageTk
import numpy as np
import torch
from torchvision import transforms, models
import tkinter as tk
from tkinter import filedialog, Label, Button

# Cargar modelo pre-entrenado para reconocimiento general
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Simulación de clases (puedes entrenar tu propio modelo para más precisión)
FISH_CLASSES = ['goldfish', 'carp', 'catfish', 'salmon', 'trout']
DISEASE_CLASSES = ['healthy', 'sick']

# Transformaciones para la imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def detect_white_spots(img_path, threshold=220, min_spots=10):
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    # Convertir a escala de grises
    gray = np.mean(img_np, axis=2)
    # Detectar píxeles "blancos"
    white_pixels = np.where(gray > threshold, 1, 0)
    # Contar agrupaciones de píxeles blancos (simulación simple)
    num_white = np.sum(white_pixels)
    # Si hay suficientes píxeles blancos, consideramos "puntos blancos"
    if num_white > min_spots * 100:  # Ajusta el factor según tus imágenes
        return True
    return False

def predict_disease(img_path):
    if detect_white_spots(img_path):
        return "Enfermo (puntos blancos detectados)"
    else:
        return "Sano"

# Interfaz gráfica simple
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        disease = predict_disease(file_path)
        result_label.config(text=f"Estado: {disease}")
        # Mostrar imagen en la interfaz
        img = Image.open(file_path)
        img = img.resize((224, 224))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

root = tk.Tk()
root.title("Detector de enfermedades en peces")

select_btn = Button(root, text="Seleccionar imagen", command=select_image)
select_btn.pack(pady=10)

result_label = Label(root, text="Resultado aparecerá aquí")
result_label.pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

root.mainloop()
