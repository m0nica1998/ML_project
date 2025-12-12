# -*- coding: utf-8 -*-

"""
SVR su MONK – Grid Search con Cross Validation
Stampa parametri ottimali, MSE medio CV, MSE Test e grafico MSE Test vs C con epsilon variabile
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
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
# 2. Grid Search SVR
# ---------------------------------------------------------

param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 0.1, 1, 10]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

svr = SVR(kernel='rbf')

grid_search = GridSearchCV(svr, param_grid, scoring=mse_scorer, cv=kf, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# 3. Risultati ottimali e MSE Test
# ---------------------------------------------------------

best_params = grid_search.best_params_
best_score_cv = -grid_search.best_score_

# Modello con parametri ottimali
svr_best = SVR(kernel='rbf', **best_params)
svr_best.fit(X_train_scaled, y_train)
y_test_pred = svr_best.predict(X_test_scaled)
best_test_mse = mean_squared_error(y_test, y_test_pred)

# Stampa dei valori richiesti
print("\n=== Risultati Ottimali ===")
print(f"Parametri ottimali: {best_params}")
print(f"MSE medio CV: {best_score_cv:.4f}")
print(f"MSE Test: {best_test_mse:.4f}")

# ---------------------------------------------------------
# 4. Calcolo MSE Test per tutte le combinazioni di parametri
# ---------------------------------------------------------

results = pd.DataFrame(grid_search.cv_results_)
results['mean_test_MSE'] = -results['mean_test_score']

test_mse_list = []
for idx, row in results.iterrows():
    svr_temp = SVR(
        kernel='rbf',
        C=row['param_C'],
        epsilon=row['param_epsilon'],
        gamma=row['param_gamma']
    )
    svr_temp.fit(X_train_scaled, y_train)
    y_pred = svr_temp.predict(X_test_scaled)
    test_mse_list.append(mean_squared_error(y_test, y_pred))

results['test_MSE'] = test_mse_list

# ---------------------------------------------------------
# 5. Grafico MSE Test vs C (epsilon variabile, gamma fisso)
# ---------------------------------------------------------

gamma_fixed = 'scale'
epsilon_values = [0.01, 0.1, 0.5]

plt.figure(figsize=(8,5))

for eps_val in epsilon_values:
    subset = results[(results['param_epsilon']==eps_val) & (results['param_gamma']==gamma_fixed)].sort_values('param_C')
    plt.plot(subset['param_C'], subset['test_MSE'], marker='o', label=f"epsilon={eps_val}")

plt.xscale('log')
plt.xlabel("C (log scale)")
plt.ylabel("MSE Test")
plt.title(f"MSE Test vs C (gamma={gamma_fixed})")
plt.legend()
plt.grid(True)
plt.show()
