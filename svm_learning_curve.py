# -*- coding: utf-8 -*-

"""
SVR su MONK – Analisi aggiuntiva: Grid Search con Cross Validation
Questo script esegue la grid search per trovare i parametri ottimali della SVR
(C, epsilon, gamma) usando 5-fold CV e mostra anche la curva di apprendimento con bande di confidenza.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, learning_curve, ShuffleSplit
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, make_scorer

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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled  = scaler.transform(X_test_encoded)

# ---------------------------------------------------------

# 2. Definizione Grid Search SVR

# ---------------------------------------------------------

param_grid = {
'C': [0.1, 1, 10, 100],
'epsilon': [0.01, 0.1, 0.5],
'gamma': ['scale', 0.1, 1, 10]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

svr = SVR(kernel='rbf')

grid_search = GridSearchCV(
svr,
param_grid,
scoring=mse_scorer,
cv=kf,
n_jobs=-1,
verbose=2
)
grid_search.fit(X_train_scaled, y_train)

# ---------------------------------------------------------

# 3. Risultati ottimali

# ---------------------------------------------------------

best_params = grid_search.best_params_
best_score = -grid_search.best_score_

print(f"\nParametri ottimali SVR: {best_params}")
print(f"MSE medio CV: {best_score:.4f}")



# ---------------------------------------------------------

# 5. Curva di apprendimento con bande di confidenza

# ---------------------------------------------------------

svr_best = SVR(kernel='rbf', **best_params)

# ShuffleSplit per ripetizioni multiple

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
svr_best,
X_train_scaled,
y_train,
cv=cv,
scoring=mse_scorer,
train_sizes=np.linspace(0.1, 1.0, 10),
n_jobs=-1
)

# Calcolo media e deviazione standard

train_mse_mean = -train_scores.mean(axis=1)
train_mse_std  = train_scores.std(axis=1)
val_mse_mean   = -val_scores.mean(axis=1)
val_mse_std    = val_scores.std(axis=1)

# Grafico con bande di confidenza

plt.figure(figsize=(10,6))
plt.plot(train_sizes, train_mse_mean, label='Training MSE', marker='o', color='blue')
plt.fill_between(train_sizes, train_mse_mean-train_mse_std, train_mse_mean+train_mse_std, color='blue', alpha=0.2)

plt.plot(train_sizes, val_mse_mean, label='Validation MSE', marker='s', color='orange')
plt.fill_between(train_sizes, val_mse_mean-val_mse_std, val_mse_mean+val_mse_std, color='orange', alpha=0.2)

plt.xlabel("Dimensione training set")
plt.ylabel("MSE")
plt.title("Curva di apprendimento SVR con bande di confidenza MONK-3")
plt.legend()
plt.grid(True)
plt.show()
