# -*- coding: utf-8 -*-
"""
KNN su MONK – Analisi aggiuntiva: K-Fold Cross Validation
Questo script esegue SOLO la cross validation per valutare accuratamente
i valori di k e selezionare il k ottimale in modo robusto.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------
# 1. Caricamento dataset
# ---------------------------------------------------------
train_file = "data/monks-3.train"
test_file  = "data/monks-3.test"

columns = ["class","A1","A2","A3","A4","A5","A6","id"]

df_train = pd.read_csv(train_file, sep=r"\s+", header=None, names=columns)
df_test  = pd.read_csv(test_file,  sep=r"\s+", header=None, names=columns)

X_train = df_train[["A1","A2","A3","A4","A5","A6"]].astype(str)
y_train = df_train["class"].astype(float)

X_test  = df_test[["A1","A2","A3","A4","A5","A6"]].astype(str)
y_test  = df_test["class"].astype(float)

encoder = OneHotEncoder(sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded  = encoder.transform(X_test)


# ---------------------------------------------------------
# 2. K-Fold Cross Validation setup
# ---------------------------------------------------------
k_values = range(1, 21)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_errors = []   # errore medio sui fold per ogni k 


# ---------------------------------------------------------
# 3. Ciclo su tutti i k
# ---------------------------------------------------------
for k in k_values:

    fold_mse = []  # errori dei 5 fold

    # K-Fold
    for train_index, val_index in kf.split(X_train_encoded):

        X_tr, X_val = X_train_encoded[train_index], X_train_encoded[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_tr, y_tr)

        pred_val = knn.predict(X_val)
        fold_mse.append(mean_squared_error(y_val, pred_val))  # errore sul vl

    # errore medio sui fold
    cv_errors.append(np.mean(fold_mse))


# ---------------------------------------------------------
# 4. Scelta del k ottimale
# ---------------------------------------------------------
optimal_k = k_values[np.argmin(cv_errors)]
print(f"\nK ottimale secondo Cross Validation: {optimal_k}")
print(f"Errore CV medio: {cv_errors[optimal_k-1]:.4f}")


# ---------------------------------------------------------
# 5. Grafico errore CV vs k
# ---------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(k_values, cv_errors, marker='o', label="CV MSE (media 5 fold)")
plt.axvline(optimal_k, color='r', linestyle='--',
            label=f"k ottimale = {optimal_k}")

plt.xlabel("Numero vicini k")
plt.ylabel("Errore MSE medio (CV)")
plt.title("K-Fold Cross Validation per KNN - MONK-3")
plt.grid(True)
plt.legend()
plt.show()
