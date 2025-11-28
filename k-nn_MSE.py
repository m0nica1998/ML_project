# -*- coding: utf-8 -*-
"""
KNN su MONK con eliminazione del rumore solo per MONK-3
Calcolo dell'errore MSE, ricerca del k ottimale
e visualizzazione dei grafici di train, validation e test.
"""

# -------------------------------
# 1. Import delle librerie
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# -------------------------------
# 2. Configurazione file
# -------------------------------
train_file = "data/monks-2.train"
test_file = "data/monks-2.test"
columns = ["class", "A1", "A2", "A3", "A4", "A5", "A6", "id"]

# -------------------------------
# 3. Caricamento dataset
# -------------------------------
df_train_full = pd.read_csv(train_file, sep=r'\s+', header=None, names=columns)
df_test = pd.read_csv(test_file, sep=r'\s+', header=None, names=columns)
print(f"Training set completo: {len(df_train_full)} esempi, Test set: {len(df_test)} esempi")

# -------------------------------
# 4. Preprocessing
# -------------------------------
X_train_full = df_train_full[["A1","A2","A3","A4","A5","A6"]].astype(str)
y_train_full = df_train_full["class"].astype(float)
X_test = df_test[["A1","A2","A3","A4","A5","A6"]].astype(str)
y_test = df_test["class"].astype(float)

encoder = OneHotEncoder(sparse_output=False)
X_train_full_encoded = encoder.fit_transform(X_train_full)
X_test_encoded = encoder.transform(X_test)

# -------------------------------
# 5. Eliminazione del rumore (solo MONK-3)
# -------------------------------
if "monks-3" in train_file.lower():
    knn_temp = KNeighborsRegressor(n_neighbors=3)
    knn_temp.fit(X_train_full_encoded, y_train_full)
    y_train_pred_temp = knn_temp.predict(X_train_full_encoded)
    errors = np.abs(y_train_full - y_train_pred_temp)
    threshold = 0.5
    mask_clean = errors <= threshold
    X_train_clean = X_train_full_encoded[mask_clean]
    y_train_clean = y_train_full[mask_clean]
    print(f"Esempi rimanenti dopo rimozione rumore: {len(y_train_clean)} / {len(y_train_full)}")
else:
    X_train_clean = X_train_full_encoded
    y_train_clean = y_train_full
    print(f"Nessun filtraggio del rumore. Numero esempi: {len(y_train_clean)}")

# -------------------------------
# 6. Suddivisione train/validation
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_train_clean, y_train_clean, test_size=0.2, random_state=42
)

# -------------------------------
# 7. Lista dei valori di k da testare
# -------------------------------
k_values = range(1, 21)
train_errors = []
val_errors = []
test_errors = []

# -------------------------------
# 8. Addestramento KNN e calcolo MSE
# -------------------------------
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_train_pred = knn.predict(X_train)
    y_val_pred = knn.predict(X_val)
    y_test_pred = knn.predict(X_test_encoded)
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    val_errors.append(mean_squared_error(y_val, y_val_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

# -------------------------------
# 9. Scelta del k ottimale
# -------------------------------
optimal_k = k_values[np.argmin(val_errors)]
print(f"\nValore ottimale di k: {optimal_k}")
print(f"Errore MSE sul test set con k ottimale: {test_errors[optimal_k-1]:.4f}")

# -------------------------------
# 10. Grafico degli errori MSE
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(k_values, train_errors, marker='o', label='Train MSE')
plt.plot(k_values, val_errors, marker='s', label='Validation MSE')
plt.plot(k_values, test_errors, marker='^', label='Test MSE')
plt.axvline(optimal_k, color='r', linestyle='--', label=f'k ottimale = {optimal_k}')
plt.xlabel('Numero di vicini k')
plt.ylabel('Errore MSE')
plt.title('Errore di Train, Validation e Test al variare di k - MONK (filtrato solo MONK-3)')
plt.legend()
plt.grid(True)
plt.show()