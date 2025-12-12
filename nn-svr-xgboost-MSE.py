import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from scipy.optimize import minimize

# ----------------------------
# 1. Caricamento dataset
# ----------------------------
train_file = "data/monks-3.train"
test_file = "data/monks-3.test"
columns = ["class","A1","A2","A3","A4","A5","A6","id"]

df_train = pd.read_csv(train_file, sep=r"\s+", header=None, names=columns)
df_test = pd.read_csv(test_file, sep=r"\s+", header=None, names=columns)

X_train_df = df_train[["A1","A2","A3","A4","A5","A6"]].astype(str)
y_train = df_train["class"].astype(float)
X_test_df  = df_test[["A1","A2","A3","A4","A5","A6"]].astype(str)
y_test = df_test["class"].astype(float)

# ----------------------------
# 2. One-hot Encoding + Scaling
# ----------------------------
encoder = OneHotEncoder(sparse_output=False)
X_train_enc = encoder.fit_transform(X_train_df)
X_test_enc  = encoder.transform(X_test_df)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enc)
X_test_scaled = scaler.transform(X_test_enc)

# ----------------------------
# 3. NN builder potenziata
# ----------------------------
def build_nn_model(units=64, lr=0.001, l2_reg=1e-6, dropout=0.1, input_dim=None):
    inp_shape = (input_dim,) if input_dim is not None else (X_train_scaled.shape[1],)
    model = keras.Sequential([
        layers.Dense(units, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg),
                     input_shape=inp_shape),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(units, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(units//2, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(1, activation="linear")
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

early_stop = callbacks.EarlyStopping(patience=20, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

# ----------------------------
# 4. Addestramento finale NN potenziata
# ----------------------------
units = 64
lr = 0.001
l2_reg = 1e-6
dropout = 0.1

final_nn = build_nn_model(units, lr, l2_reg, dropout, input_dim=X_train_scaled.shape[1])

history = final_nn.fit(
    X_train_scaled, y_train,
    epochs=400, batch_size=16,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop, reduce_lr]
)

nn_pred_test = final_nn.predict(X_test_scaled).flatten()
nn_mse = mean_squared_error(y_test, nn_pred_test)

# ----------------------------
# 5. SVR con GridSearch
# ----------------------------
svr_grid = {
    "C":[1,10,50],
    "epsilon":[0.01,0.05,0.1],
    "gamma":["scale","auto"],
    "kernel":["rbf"]
}

svr = GridSearchCV(SVR(), svr_grid, cv=4, scoring="neg_mean_squared_error", n_jobs=-1)
svr.fit(X_train_scaled, y_train)
best_svr = svr.best_estimator_

svr_pred_test = best_svr.predict(X_test_scaled)
svr_mse = mean_squared_error(y_test, svr_pred_test)

# ----------------------------
# 6. XGBoost
# ----------------------------
xgb = XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0
)
xgb.fit(X_train_scaled, y_train)
xgb_pred_test = xgb.predict(X_test_scaled)
xgb_mse = mean_squared_error(y_test, xgb_pred_test)

# ----------------------------
# 7. Ensemble ottimizzato NN + SVR + XGB
# ----------------------------
X_tr_full, X_val_full, y_tr_full, y_val_full = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

nn_val_pred = final_nn.predict(X_val_full).flatten()
svr_val_pred = best_svr.predict(X_val_full)
xgb_val_pred = xgb.predict(X_val_full)

preds_list = [nn_val_pred, svr_val_pred, xgb_val_pred]

def ensemble_mse_loss(weights, preds_list, y_true):
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # normalize to sum=1
    ens_pred = sum(w*p for w,p in zip(weights, preds_list))
    return mean_squared_error(y_true, ens_pred)

bounds = [(0,1)]*3
constraints = {'type':'eq', 'fun': lambda w: np.sum(w)-1}

res = minimize(ensemble_mse_loss, x0=[1/3]*3, args=(preds_list, y_val_full),
               bounds=bounds, constraints=constraints)
alpha_nn, alpha_svr, alpha_xgb = res.x
alpha_nn, alpha_svr, alpha_xgb = alpha_nn/np.sum(res.x), alpha_svr/np.sum(res.x), alpha_xgb/np.sum(res.x)

ensemble_pred_test = alpha_nn * nn_pred_test + alpha_svr * svr_pred_test + alpha_xgb * xgb_pred_test
ensemble_mse = mean_squared_error(y_test, ensemble_pred_test)

# ----------------------------
# 8. Stampa risultati
# ----------------------------
print(f"MSE NN test: {nn_mse:.4f}")
print(f"MSE SVR test: {svr_mse:.4f}")
print(f"MSE XGBoost test: {xgb_mse:.4f}")
print(f"Alpha ottimali ensemble -> NN: {alpha_nn:.2f}, SVR: {alpha_svr:.2f}, XGB: {alpha_xgb:.2f}")
print(f"MSE Ensemble test: {ensemble_mse:.4f}")

# ----------------------------
# 9. Plot learning curve NN (Training vs Validation)
# ----------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training MSE")
plt.plot(history.history["val_loss"], label="Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Learning Curve NN MONK-3")
plt.grid(True)
plt.legend()
plt.show()
