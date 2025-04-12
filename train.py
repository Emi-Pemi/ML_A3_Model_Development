import os
import numpy as np
import joblib
import wandb
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === Custom Linear Regression with Gradient Descent ===
class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000, l1=0.0, l2=0.01, early_stopping_rounds=10, tol=1e-4):
        self.lr = lr
        self.epochs = epochs
        self.l1 = l1
        self.l2 = l2
        self.early_stopping_rounds = early_stopping_rounds
        self.tol = tol

    def fit(self, X, y, X_val=None, y_val=None):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0

        best_loss = float("inf")
        no_improve = 0

        for epoch in range(self.epochs):
            y_pred = X.dot(self.w) + self.b
            error = y_pred - y

            dw = (X.T.dot(error) / m) + self.l2 * self.w + self.l1 * np.sign(self.w)
            db = np.mean(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = mean_squared_error(y_val, val_pred)

                print(f"Epoch {epoch}: val_loss = {val_loss:.4f}")
                if best_loss - val_loss > self.tol:
                    best_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self.early_stopping_rounds:
                    print(f"Early stopping at epoch {epoch}")
                    break

    def predict(self, X):
        return X.dot(self.w) + self.b

# === W&B Setup ===
os.environ["WANDB_DISABLE_SYMLINKS"] = "true"
wandb.login(key="448d91413c277f620a34c7d1981dd9d03008d287")
wandb.init(project="california-housing-optimized", name="LinearRegressionGD")

# === Load and Prepare Data ===
data = fetch_california_housing()
X, y = data.data, data.target
y = y * 100000
y = np.clip(y, 50000, 500000)
y_trans = np.log(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_trans, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# === Train Custom Model ===
model = LinearRegressionGD(lr=0.01, epochs=1000, l1=0.0, l2=0.01, early_stopping_rounds=10)
model.fit(X_train_scaled, y_train, X_val_scaled, y_val)

# === Evaluation ===
def evaluate(X, y_true_trans):
    y_pred_trans = model.predict(X)
    y_pred = np.exp(y_pred_trans)
    y_true = np.exp(y_true_trans)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2, y_pred

val_mse, val_r2, val_preds = evaluate(X_val_scaled, y_val)
test_mse, test_r2, test_preds = evaluate(X_test_scaled, y_test)

# === Save Model ===
model_data = {
    'model_weights': model.w.tolist(),
    'model_bias': model.b,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'transformation': 'log',
    'feature_names': data.feature_names,
    'input_ranges': {
        'min': X.min(axis=0).tolist(),
        'max': X.max(axis=0).tolist()
    },
    'test_performance': {
        'mse': test_mse,
        'r2': test_r2
    }
}
joblib.dump(model_data, 'model.joblib')

# === W&B Logging ===
wandb.log({
    'validation_mse': val_mse,
    'validation_r2': val_r2,
    'test_mse': test_mse,
    'test_r2': test_r2,
    'validation_samples': wandb.Table(
        columns=["True Value", "Predicted Value"],
        data=list(zip(np.exp(y_val)[:100], val_preds[:100]))
    )
})
wandb.finish()

# === Final Output ===
print(f"""
=== Custom Linear Regression with GD ===
Validation MSE: {val_mse:.4f} (${np.sqrt(val_mse):,.2f} error)
Validation R²: {val_r2:.4f}
Test MSE: {test_mse:.4f} (${np.sqrt(test_mse):,.2f} error)
Test R²: {test_r2:.4f}

Model saved as model.joblib
""")
