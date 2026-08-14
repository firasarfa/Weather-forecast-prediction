import os
# Suppress TensorFlow info/warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lstm_preparation import prepare_lstm_data, FEATURE_COLUMNS, TARGET_COL
from model import LEARNING_RATE

def inverse_transform_target(pred_or_true, scaler):
    """
    Helper to inverse transform only the target column when scaler fits on multiple features.
    """
    dummy = np.zeros((len(pred_or_true), len(FEATURE_COLUMNS)))
    target_idx = FEATURE_COLUMNS.index(TARGET_COL)
    dummy[:, target_idx] = pred_or_true.flatten()
    inverse = scaler.inverse_transform(dummy)
    return inverse[:, target_idx]

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluates the model and plots detailed diagnostic results."""
    print("Starting comprehensive evaluation...")
    
    # 1. Make Predictions
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # 2. Inverse Scale
    y_pred = inverse_transform_target(y_pred_scaled, scaler)
    y_true = inverse_transform_target(y_test, scaler)
    
    # 3. Calculate Metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Naive Baseline (Persistence: y_t = y_t-1)
    naive_mae = mean_absolute_error(y_true[1:], y_true[:-1])
    
    print(f"\n*** Performance Summary ***")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f} °C")
    print(f"R² Score: {r2:.4f}")
    print(f"Naive Baseline MAE: {naive_mae:.4f} °C")
    print(f"Improvement over Baseline: {((naive_mae - mae) / naive_mae * 100):.2f}%")

    # 4. Visualization
    fig = plt.figure(figsize=(16, 12))
    
    # A. Time Series Forecast (Zoomed In)
    plt.subplot(2, 2, 1)
    plt.plot(y_true[:200], label='Actual', color='#1f77b4', linewidth=1.5)
    plt.plot(y_pred[:200], label='Predicted', color='#ff7f0e', linestyle='--', alpha=0.8)
    plt.title('Time Series Forecast (First 200 Hours)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # B. Scatter Plot: Actual vs. Predicted
    plt.subplot(2, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.3, s=10, color='teal')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(f'Actual vs. Predicted (R² = {r2:.3f})')
    plt.xlabel('Actual Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.grid(True, alpha=0.3)

    # C. Residual Distribution (Error Analysis)
    residuals = y_true - y_pred
    plt.subplot(2, 2, 3)
    sns.histplot(residuals, kde=True, color='purple', bins=40)
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f'Residuals Distribution (Mean: {np.mean(residuals):.3f})')
    plt.xlabel('Prediction Error (°C)')
    plt.ylabel('Frequency')

    # D. Error over Time (Absolute Error)
    plt.subplot(2, 2, 4)
    plt.plot(np.abs(residuals)[:500], color='crimson', alpha=0.6)
    plt.title('Absolute Error over Time (First 500 Hours)')
    plt.ylabel('Error (°C)')
    plt.xlabel('Time (Hours)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 5. Top 5 Worst Predictions Analysis
    abs_err = np.abs(residuals)
    worst_indices = np.argsort(abs_err)[-5:][::-1]
    
    print("\n--- Top 5 Worst Predictions ---")
    for idx in worst_indices:
        print(f"Index: {idx:<5} | Error: {residuals[idx]:.2f}°C | Actual: {y_true[idx]:.2f}°C | Pred: {y_pred[idx]:.2f}°C")

if __name__ == '__main__':
    print("Loading data and model for evaluation...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_lstm_data()
    
    try:
        model = load_model('best_rnn_model.keras')
        evaluate_model(model, X_test, y_test, scaler)
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Make sure you have run the training script (model.py) first.")