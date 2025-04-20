# Código completo actualizado con integración de TensorFlow para predecir el GRD,
# visualización, matriz de confusión y guardado de gráficos.

import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import math

# --- 1. Cargar dataset base ---
corpus = []
target = []
edad = []
sexo = []

with open("dataset_elpino.csv", encoding="utf-8") as archivo:
    header = archivo.readline().strip().split(";")
    features = []
    for col in header:
        col = col.split("-")[0].strip()
        if col.startswith("Diag") or col.startswith("Proc"):
            col = col.split(" ")
            col = col[0] + col[1]
        features.append(col)

    for linea in archivo:
        row = []
        linea = linea.strip().split(";")
        for i in range(len(linea)):
            col = linea[i].split("-")[0].strip()
            if i == 67:
                target.append(col)
            elif i == 66:
                sexo.append(1 if col == "Mujer" else 0)
            elif i == 65:
                edad.append(int(col))
            else:
                row.append(col)
        corpus.append(row)

df = pd.DataFrame(corpus, columns=features[:-3])
df["Edad"] = edad
df["Sexo"] = sexo
df["GRD"] = target

# --- 2. Cargar CIE-10, CIE-9 y GRD ---
cie10_df = pd.read_excel("cie10.xlsx")
cie9_df = pd.read_excel("cie9.xlsx")
grd_maestras = pd.read_excel("tmb.xlsx", sheet_name=None)
grd_df = pd.concat(grd_maestras.values(), ignore_index=True)

# --- 3. Descripciones CIE-10 para diagnósticos ---
cie10_map = cie10_df.set_index("Código")["Descripción"].to_dict()
for j in range(1, 36):
    col = f"Diag{str(j).zfill(2)}"
    nueva_col = f"{col}_desc"
    df[nueva_col] = df[col].map(cie10_map)

# --- 4. Descripciones CIE-9 para procedimientos ---
cie9_map = cie9_df.set_index("Código")["Descripción"].to_dict()
for j in range(1, 31):
    col = f"Proced{str(j).zfill(2)}"
    nueva_col = f"{col}_desc"
    df[nueva_col] = df[col].map(cie9_map)

# --- 5. Enlazar GRD con severidad y mortalidad ---
grd_df_renamed = grd_df.rename(columns={"IR- GRD": "GRD"}).dropna(subset=["GRD"])
grd_df_renamed["GRD"] = grd_df_renamed["GRD"].astype(str).str.strip()
df["GRD"] = df["GRD"].astype(str).str.strip()
df = df.merge(grd_df_renamed[["GRD", "Severidad", "Mortalidad"]], on="GRD", how="left")

# --- 6. Visualizaciones (gráficos guardados como PNG) ---
if df["GRD"].notna().sum() > 0:
    plt.figure(figsize=(10, 5))
    df["GRD"].value_counts().head(10).plot(kind="bar")
    plt.title("Top 10 GRDs más frecuentes")
    plt.ylabel("Cantidad de casos")
    plt.xlabel("GRD")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("top_grd.png")
    plt.close()

if df["Severidad"].notna().sum() > 0:
    plt.figure(figsize=(6, 4))
    df["Severidad"].value_counts().sort_index().plot(kind="bar", color="skyblue")
    plt.title("Distribución de Severidad")
    plt.ylabel("Cantidad")
    plt.xlabel("Nivel de Severidad")
    plt.tight_layout()
    plt.savefig("distribucion_severidad.png")
    plt.close()

if df["Mortalidad"].notna().sum() > 0:
    plt.figure(figsize=(6, 4))
    df["Mortalidad"].value_counts().sort_index().plot(kind="bar", color="salmon")
    plt.title("Distribución de Mortalidad")
    plt.ylabel("Cantidad")
    plt.xlabel("Nivel de Mortalidad")
    plt.tight_layout()
    plt.savefig("distribucion_mortalidad.png")
    plt.close()

if df["Diag01_desc"].notna().sum() > 0:
    plt.figure(figsize=(10, 5))
    df["Diag01_desc"].value_counts().head(10).plot(kind="bar")
    plt.title("Top 10 Diagnósticos principales (Diag01)")
    plt.ylabel("Frecuencia")
    plt.xlabel("Descripción")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("top_diag01.png")
    plt.close()

if df["Proced01_desc"].notna().sum() > 0:
    plt.figure(figsize=(10, 5))
    df["Proced01_desc"].value_counts().head(10).plot(kind="bar", color="orange")
    plt.title("Top 10 Procedimientos principales (Proced01)")
    plt.ylabel("Frecuencia")
    plt.xlabel("Descripción")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("top_proced01.png")
    plt.close()

plt.figure(figsize=(8, 6))
corr_matrix = df[["Edad", "Sexo"]].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Matriz de correlación")
plt.tight_layout()
plt.savefig("matriz_correlacion.png")
plt.close()

# --- 7. Modelo de predicción con TensorFlow ---
diag_cols = [f"Diag{str(i).zfill(2)}" for i in range(1, 36)]
proc_cols = [f"Proced{str(i).zfill(2)}" for i in range(1, 31)]
numeric_cols = ["Edad", "Sexo"]
feature_cols = diag_cols + proc_cols + numeric_cols

df_modelo = df[feature_cols + ["GRD"]].copy()
df_modelo[diag_cols + proc_cols] = df_modelo[diag_cols + proc_cols].fillna("UNK")

label_encoders = {}
for col in diag_cols + proc_cols:
    le = LabelEncoder()
    df_modelo[col] = le.fit_transform(df_modelo[col])
    label_encoders[col] = le

le_grd = LabelEncoder()
df_modelo["GRD_encoded"] = le_grd.fit_transform(df_modelo["GRD"])
y = to_categorical(df_modelo["GRD_encoded"])
X = df_modelo[diag_cols + proc_cols + numeric_cols].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
loss, acc = model.evaluate(X_test, y_test, verbose=0)

# --- 8. Guardar gráfico de accuracy y loss ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Evolución de la Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Evolución del Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("entrenamiento_accuracy_loss.png")
plt.close()

# --- 9. Matrices de Confusión por bloques de GRD ---



# Predicciones
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Obtener GRD presentes en test
clases_presentes = np.unique(np.concatenate((y_true, y_pred)))
etiquetas_presentes = le_grd.inverse_transform(clases_presentes)

# Parámetros del bloque
bloque_tamano = 30  # puedes cambiar a 20, 50, etc.
total_bloques = math.ceil(len(clases_presentes) / bloque_tamano)

for i in range(total_bloques):
    inicio = i * bloque_tamano
    fin = (i + 1) * bloque_tamano
    ids_bloque = clases_presentes[inicio:fin]
    etiquetas_bloque = le_grd.inverse_transform(ids_bloque)

    cm = confusion_matrix(y_true, y_pred, labels=ids_bloque)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquetas_bloque)

    plt.figure(figsize=(12, 10))
    disp.plot(xticks_rotation='vertical', cmap='Blues')
    plt.title(f"Matriz de Confusión - GRD Bloque {i+1}")
    plt.tight_layout()
    plt.savefig(f"matriz_confusion_bloque_{i+1}.png")
    plt.close()


# --- 10. Guardar el modelo entrenado en archivo .h5 ---
modelo_guardado = "modelo_grd.h5"
model.save(modelo_guardado)
print(f"✅ Modelo guardado exitosamente como: {modelo_guardado}")

# --- 11. Generar archivo requirements.txt ---
import pkg_resources

with open("requirements.txt", "w") as f:
    for dist in pkg_resources.working_set:
        f.write(f"{dist.project_name}=={dist.version}\n")

print("✅ Archivo requirements.txt generado con éxito.")
