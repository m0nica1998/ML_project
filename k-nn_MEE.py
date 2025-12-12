# -*- coding: utf-8 -*-
"""
K-NN regression su MONK (lettura .train/.test), filtraggio rumore solo per MONK-3, e analisi MEE.
- Calcolo MEE come media delle distanze euclidee punto-per-punto:
    MEE = (1 / l) * sum_{p=1..l} || o_p - t_p ||_2
- Trova il k ottimale usando il validation set (splitting del .train).
- Grafico delle curve Train / Validation / Test (MEE).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# ---------------------------------------------------------------------
# CONFIGURAZIONE FILE
# ---------------------------------------------------------------------
train_file = "data/monks-3.train"  # Cambia per MONK-1 o MONK-2
test_file  = "data/monks-3.test"

columns = ["class", "A1", "A2", "A3", "A4", "A5", "A6", "id"]

# ---------------------------------------------------------------------
# 1) LETTURA DEI FILE .train E .test
# ---------------------------------------------------------------------
df_train_full = pd.read_csv(train_file, sep=r'\s+', header=None, names=columns)
df_test       = pd.read_csv(test_file,  sep=r'\s+', header=None, names=columns)

print(f"Training full: {len(df_train_full)} righe, Test: {len(df_test)} righe")

# ---------------------------------------------------------------------
# 2) PREPROCESSING: feature/target e one-hot encoding
# ---------------------------------------------------------------------
X_train_full = df_train_full[["A1","A2","A3","A4","A5","A6"]].astype(str)
y_train_full = df_train_full["class"].astype(float)

X_test = df_test[["A1","A2","A3","A4","A5","A6"]].astype(str)
y_test = df_test["class"].astype(float)

encoder = OneHotEncoder(sparse_output=False)
X_train_full_encoded = encoder.fit_transform(X_train_full)
X_test_encoded       = encoder.transform(X_test)

# ---------------------------------------------------------------------
# 3) RIMOZIONE DEL RUMORE SOLO SE MONK-3
# ---------------------------------------------------------------------
if "monks-3" in train_file.lower():
    # Predizione preliminare con k piccolo (3)
    knn_temp = KNeighborsRegressor(n_neighbors=3)
    knn_temp.fit(X_train_full_encoded, y_train_full)
    y_train_pred_temp = knn_temp.predict(X_train_full_encoded)

    # Calcolo errore assoluto
    abs_errors = np.abs(y_train_full - y_train_pred_temp)

    # Filtro soglia per rumore
    threshold = 0.5
    mask_clean = abs_errors <= threshold

    X_train_clean = X_train_full_encoded[mask_clean]
    y_train_clean = y_train_full[mask_clean]

    print(f"Esempi rimanenti dopo filtraggio rumore: {len(y_train_clean)} / {len(y_train_full)}")
else:
    # Non filtrare gli esempi per MONK-1 e MONK-2
    X_train_clean = X_train_full_encoded
    y_train_clean = y_train_full
    print(f"Nessun filtraggio del rumore applicato. Numero esempi: {len(y_train_clean)}")

# ---------------------------------------------------------------------
# 4) DIVISIONE IN TRAIN E VALIDATION
# ---------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_train_clean, y_train_clean, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------
# 5) FUNZIONE PER LA MEE (Mean Euclidean Error)
# ---------------------------------------------------------------------
def mean_euclidean_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    per_sample_l2 = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    return np.mean(per_sample_l2)

# ---------------------------------------------------------------------
# 6) GRID SEARCH SU k: calcolo errori MEE
# ---------------------------------------------------------------------
k_values = range(1, 21)
train_errors = []
val_errors   = []
test_errors  = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_train_pred = knn.predict(X_train)
    y_val_pred   = knn.predict(X_val)
    y_test_pred  = knn.predict(X_test_encoded)

    train_errors.append(mean_euclidean_error(y_train, y_train_pred))
    val_errors.append(mean_euclidean_error(y_val,   y_val_pred))
    test_errors.append(mean_euclidean_error(y_test,  y_test_pred))

# ---------------------------------------------------------------------
# 7) Scelta del k ottimale
# ---------------------------------------------------------------------
optimal_k = k_values[np.argmin(val_errors)]
print(f"\nK ottimale (min val MEE): {optimal_k}")
print(f"MEE sul test set con k={optimal_k}: {test_errors[optimal_k-1]:.4f}")

# ---------------------------------------------------------------------
# 8) GRAFICO CURVE MEE
# ---------------------------------------------------------------------
plt.figure(figsize=(9,6))
plt.plot(k_values, train_errors, marker='o', label='Train MEE')
plt.plot(k_values, val_errors,   marker='s', label='Validation MEE')
plt.plot(k_values, test_errors,  marker='^', label='Test MEE')
plt.axvline(optimal_k, color='red', linestyle='--', label=f'k ottimale = {optimal_k}')

plt.xlabel('Numero di vicini k')
plt.ylabel('MEE (Mean Euclidean Error)')
plt.title('MEE: Train / Validation / Test al variare di k (MONK-3)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
