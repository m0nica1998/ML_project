# -*- coding: utf-8 -*-
"""
KNN su MONK con eliminazione del rumore solo per MONK-3
Analisi Bias-Variance al variare di k, insieme a MSE.
Ora incluso anche il grafico con Bias^2 + Varianza.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------
# 1. Configurazione dataset
# ---------------------------------------------------------
train_file = "data/monks-3.train"
test_file  = "data/monks-3.test"

columns = ["class","A1","A2","A3","A4","A5","A6","id"]


# ---------------------------------------------------------
# 2. Caricamento dataset
# ---------------------------------------------------------
df_train_full = pd.read_csv(train_file, sep=r"\s+", header=None, names=columns)
df_test       = pd.read_csv(test_file,  sep=r"\s+", header=None, names=columns)

print(f"Training set completo: {len(df_train_full)}, Test set: {len(df_test)}")


# ---------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------
X_train_full = df_train_full[["A1","A2","A3","A4","A5","A6"]].astype(str)
y_train_full = df_train_full["class"].astype(float)

X_test = df_test[["A1","A2","A3","A4","A5","A6"]].astype(str)
y_test = df_test["class"].astype(float)

encoder = OneHotEncoder(sparse_output=False)
X_train_full_encoded = encoder.fit_transform(X_train_full)
X_test_encoded       = encoder.transform(X_test)


# ---------------------------------------------------------
# 4. Eliminazione del rumore (solo MONK-3)
# ---------------------------------------------------------
if "monks-3" in train_file.lower():
    knn_temp = KNeighborsRegressor(n_neighbors=3)
    knn_temp.fit(X_train_full_encoded, y_train_full)
    y_pred_temp = knn_temp.predict(X_train_full_encoded)

    abs_err = np.abs(y_train_full - y_pred_temp)
    threshold = 0.5
    mask_clean = abs_err <= threshold

    X_train_clean = X_train_full_encoded[mask_clean]
    y_train_clean = y_train_full[mask_clean]

    print(f"Esempi rimanenti dopo filtraggio rumore: {len(y_train_clean)} / {len(y_train_full)}")
else:
    X_train_clean = X_train_full_encoded
    y_train_clean = y_train_full
    print(f"Nessun filtraggio del rumore. Numero esempi: {len(y_train_clean)}")


# ---------------------------------------------------------
# 5. Train/Validation split
# ---------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_train_clean, y_train_clean, test_size=0.2, random_state=42
)


# ---------------------------------------------------------
# 6. Inizializzazione variabili
# ---------------------------------------------------------
k_values = range(1, 21)
train_errors = []
val_errors   = []
test_errors  = []
bias2_list   = []
var_list     = []

n_bootstrap = 50


# ---------------------------------------------------------
# 7. Loop su k: MSE + Bias^2 + Varianza
# ---------------------------------------------------------
for k in k_values:

    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)

    # MSE
    train_errors.append(mean_squared_error(y_train, knn.predict(X_train)))
    val_errors.append(mean_squared_error(y_val, knn.predict(X_val)))
    test_errors.append(mean_squared_error(y_test, knn.predict(X_test_encoded)))

    # ---------------------------------------------------------
    # Stima Bias^2 e Varianza (bootstrap)
    # ---------------------------------------------------------
    val_preds_boot = np.zeros((len(y_val), n_bootstrap))

    for b in range(n_bootstrap):
        idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        Xb, yb = X_train[idx], y_train.iloc[idx] if isinstance(y_train, pd.Series) else y_train[idx]

        knn_b = KNeighborsRegressor(n_neighbors=k)
        knn_b.fit(Xb, yb)

        val_preds_boot[:, b] = knn_b.predict(X_val)

    pred_mean = np.mean(val_preds_boot, axis=1)
    pred_var  = np.var(val_preds_boot, axis=1)
    bias2     = (pred_mean - y_val)**2

    bias2_list.append(np.mean(bias2))
    var_list.append(np.mean(pred_var))


# ---------------------------------------------------------
# 8. Trovare k ottimale
# ---------------------------------------------------------
optimal_k = k_values[np.argmin(val_errors)]
optimal_mse = val_errors[optimal_k-1]

print(f"\nValore ottimale k: {optimal_k}")
print(f"MSE validation al k ottimale: {optimal_mse:.4f}")


# ---------------------------------------------------------
# 9. Plot Bias^2, Varianza e Bias^2 + Varianza
# ---------------------------------------------------------
plt.figure(figsize=(9,6))

plt.plot(k_values, bias2_list, marker='o', label="$Bias^2$")
plt.plot(k_values, var_list, marker='s', label="Varianza")

# NUOVA CURVA: Bias^2 + Varianza
sum_curve = np.array(bias2_list) + np.array(var_list)
plt.plot(k_values, sum_curve, marker='^', linestyle='--', label="$Bias^2 + Varianza$")

plt.axvline(optimal_k, color='r', linestyle='--', label=f"k ottimale = {optimal_k}")

plt.xlabel("Numero di vicini k")
plt.ylabel("Valore")
plt.title("Bias$^2$, Varianza e Bias$^2$+Varianza vs k - MONK-3")
plt.grid(True)
plt.legend()
plt.show()
