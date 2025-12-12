import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from scipy.optimize import minimize

# ---------------------------------------------------------------------
# FUNZIONE PER LA MEE (Mean Euclidean Error)
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
# Callback custom per MEE
# ----------------------------
class MEECallback(keras.callbacks.Callback):
    def __init__(self, train_data, val_data):
        super().__init__()
        self.X_train, self.y_train = train_data
        self.X_val, self.y_val = val_data
        self.train_mee = []
        self.val_mee = []

    def on_epoch_end(self, epoch, logs=None):
        train_pred = self.model.predict(self.X_train, verbose=0).flatten()
        val_pred = self.model.predict(self.X_val, verbose=0).flatten()
        self.train_mee.append(mean_euclidean_error(self.y_train, train_pred))
        self.val_mee.append(mean_euclidean_error(self.y_val, val_pred))

# ----------------------------
# 4. Addestramento finale NN potenziata
# ----------------------------
units = 64
lr = 0.001
l2_reg = 1e-6
dropout = 0.1

final_nn = build_nn_model(units, lr, l2_reg, dropout, input_dim=X_train_scaled.shape[1])

# Split training/validation per callback MEE
X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)
mee_callback = MEECallback((X_train_sub, y_train_sub), (X_val_sub, y_val_sub))

history = final_nn.fit(
    X_train_sub, y_train_sub,
    validation_data=(X_val_sub, y_val_sub),
    epochs=400, batch_size=16,
    verbose=0,
    callbacks=[early_stop, reduce_lr, mee_callback]
)

nn_pred_test = final_nn.predict(X_test_scaled).flatten()
nn_mee = mean_euclidean_error(y_test, nn_pred_test)

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
svr_mee = mean_euclidean_error(y_test, svr_pred_test)

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
xgb_mee = mean_euclidean_error(y_test, xgb_pred_test)

# ----------------------------
# 7. Ensemble ottimizzato NN + SVR + XGB
# ----------------------------
nn_val_pred = final_nn.predict(X_val_sub).flatten()
svr_val_pred = best_svr.predict(X_val_sub)
xgb_val_pred = xgb.predict(X_val_sub)

preds_list = [nn_val_pred, svr_val_pred, xgb_val_pred]

def ensemble_mee_loss(weights, preds_list, y_true):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    ens_pred = sum(w*p for w,p in zip(weights, preds_list))
    return mean_euclidean_error(y_true, ens_pred)

bounds = [(0,1)]*3
constraints = {'type':'eq', 'fun': lambda w: np.sum(w)-1}

res = minimize(ensemble_mee_loss, x0=[1/3]*3, args=(preds_list, y_val_sub),
               bounds=bounds, constraints=constraints)
alpha_nn, alpha_svr, alpha_xgb = res.x
alpha_nn, alpha_svr, alpha_xgb = alpha_nn/np.sum(res.x), alpha_svr/np.sum(res.x), alpha_xgb/np.sum(res.x)

ensemble_pred_test = alpha_nn * nn_pred_test + alpha_svr * svr_pred_test + alpha_xgb * xgb_pred_test
ensemble_mee = mean_euclidean_error(y_test, ensemble_pred_test)

# ----------------------------
# 8. Stampa risultati
# ----------------------------
print(f"MEE NN test: {nn_mee:.4f}")
print(f"MEE SVR test: {svr_mee:.4f}")
print(f"MEE XGBoost test: {xgb_mee:.4f}")
print(f"Alpha ottimali ensemble -> NN: {alpha_nn:.2f}, SVR: {alpha_svr:.2f}, XGB: {alpha_xgb:.2f}")
print(f"MEE Ensemble test: {ensemble_mee:.4f}")

# ----------------------------
# 9. Plot learning curve NN (Training vs Validation) con MEE
# ----------------------------
plt.figure(figsize=(8,5))
plt.plot(mee_callback.train_mee, label="Training MEE")
plt.plot(mee_callback.val_mee, label="Validation MEE")
plt.xlabel("Epoch")
plt.ylabel("MEE")
plt.title("Learning Curve NN MONK-3")
plt.grid(True)
plt.legend()
plt.show()
