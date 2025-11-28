# -*- coding: utf-8 -*-
"""
Analisi approfondita di KNN sul dataset MONK
Include:
1. Sensibilità agli attributi (Leave-One-Feature-Out)
2. Analisi delle distanze tra punti
3. Curva di apprendimento
4. Sensibilità al rumore
5. Analisi dei vicini
6. Distribuzione degli errori
7. Confronto pesi vicini (uniform vs distance)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# -------------------------------
# Configurazione e lettura dataset
# -------------------------------
train_file = "data/monks-3.train"
test_file  = "data/monks-3.test"
columns = ["class","A1","A2","A3","A4","A5","A6","id"]

df_train = pd.read_csv(train_file, sep=r"\s+", header=None, names=columns)
df_test  = pd.read_csv(test_file, sep=r"\s+", header=None, names=columns)

# Selezione features e target
features = ["A1","A2","A3","A4","A5","A6"]
X_train = df_train[features].astype(str)
y_train = df_train["class"].astype(float)
X_test  = df_test[features].astype(str)
y_test  = df_test["class"].astype(float)

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False)
X_train_enc = encoder.fit_transform(X_train)
X_test_enc  = encoder.transform(X_test)

# Split train/validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train_enc, y_train, test_size=0.2, random_state=42)



# -------------------------------
# Curva di apprendimento
# -------------------------------
print("\nCurva di apprendimento")
train_sizes = np.linspace(0.1, 1.0, 10)
train_errors = []
val_errors = []

for frac in train_sizes:
    n_samples = int(frac * len(X_tr))
    X_sub = X_tr[:n_samples]
    y_sub = y_tr.iloc[:n_samples]
    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(X_sub, y_sub)
    train_errors.append(mean_squared_error(y_sub, knn.predict(X_sub)))
    val_errors.append(mean_squared_error(y_val, knn.predict(X_val)))

plt.figure(figsize=(7,5))
plt.plot(train_sizes*len(X_tr), train_errors, marker='o', label='Train MSE')
plt.plot(train_sizes*len(X_tr), val_errors, marker='s', label='Validation MSE')
plt.xlabel("Numero esempi training")
plt.ylabel("MSE")
plt.title("Curva di apprendimento KNN MONK-3")
plt.legend()
plt.grid(True)
plt.show()







