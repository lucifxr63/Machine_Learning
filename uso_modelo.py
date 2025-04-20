#ejemplo de uso de modelo

from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

# Cargar modelo
modelo = load_model("modelo_grd.h5")

# Simular una nueva entrada (reemplazar con datos reales preprocesados)
entrada = np.random.randint(0, 500, size=(1, 67))  # 67 columnas como en el entrenamiento

# Predicción
pred = modelo.predict(entrada)
clase_predicha = np.argmax(pred, axis=1)
print("GRD predicho (índice):", clase_predicha)
print("GRD predicho (valor):", clase_predicha[0] * 0.01)  # Convertir índice a valor real
# Convertir a DataFrame para guardar en CSV