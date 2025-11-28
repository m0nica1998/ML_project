# -*- coding: utf-8 -*-

"""
SVR su MONK – Grid Search + curve 1D MSE vs C, epsilon, gamma
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.metrics import  make_scorer

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

# ---------------------------------------------------------------------
#  FUNZIONE PER LA MEE (Mean Euclidean Error)
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


# ---------------------------------------------------------
# 2. Grid Search SVR
# ---------------------------------------------------------

param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 0.1, 1, 10]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scorer = make_scorer(mean_euclidean_error, greater_is_better=False)

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
# 3. Risultati
# ---------------------------------------------------------

results = pd.DataFrame(grid_search.cv_results_)
results["mean_test_MSE"] = -results["mean_test_score"]

best_params = grid_search.best_params_
print("\nParametri ottimali:", best_params)

# ---------------------------------------------------------
# 4. Grafici richiesti — curve 1D
# ---------------------------------------------------------

# Fissiamo gli iperparametri migliori
C_best = best_params["C"]
eps_best = best_params["epsilon"]
gamma_best = best_params["gamma"]

# ---------------------------------------------------------
# 4.1 MEE vs C (epsilon e gamma fissi)
# ---------------------------------------------------------

subset_C = results[
    (results["param_epsilon"] == eps_best) &
    (results["param_gamma"] == gamma_best)
].sort_values("param_C")

plt.figure(figsize=(8,5))
plt.plot(subset_C["param_C"], subset_C["mean_test_MSE"], marker='o')
plt.xscale('log')
plt.xlabel("C (log scale)")
plt.ylabel("MEE CV")
plt.title(f"MEE vs C (epsilon={eps_best}, gamma={gamma_best}) MONK-3")
plt.grid(True)
plt.show()

# ---------------------------------------------------------
# 4.2 MEE vs epsilon (C e gamma fissi)
# ---------------------------------------------------------

subset_eps = results[
    (results["param_C"] == C_best) &
    (results["param_gamma"] == gamma_best)
].sort_values("param_epsilon")

plt.figure(figsize=(8,5))
plt.plot(subset_eps["param_epsilon"], subset_eps["mean_test_MSE"], marker='s')
plt.xlabel("epsilon")
plt.ylabel("MEE CV")
plt.title(f"MEE vs epsilon (C={C_best}, gamma={gamma_best}) MONK-3")
plt.grid(True)
plt.show()

# ---------------------------------------------------------
# 4.3 MEE vs gamma (C e epsilon fissi)
# ---------------------------------------------------------

subset_gamma = results[
    (results["param_C"] == C_best) &
    (results["param_epsilon"] == eps_best)
].copy()

subset_gamma["gamma_str"] = subset_gamma["param_gamma"].astype(str)

# ordiniamo: "scale" per primo, poi numeri
subset_gamma = subset_gamma.sort_values(
    by="gamma_str",
    key=lambda x: x.map(lambda v: -1 if v=="scale" else float(v))
)

plt.figure(figsize=(8,5))
plt.plot(subset_gamma["gamma_str"], subset_gamma["mean_test_MSE"], marker='^')
plt.xlabel("gamma")
plt.ylabel("MEE CV")
plt.title(f"MEE vs gamma (C={C_best}, epsilon={eps_best}) MONK-3")
plt.grid(True)
plt.show()
