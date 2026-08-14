import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_prep import load_and_preprocess_data

# --- KONSTANTEN ---
TARGET_COL = 'T (degC)'
TIME_STEPS = 72          # 3 Tage Kontext
FORECAST_HORIZON = 1     # Vorhersage für t+1
# Wir definieren die Splits für die spätere Aufteilung der Sequenzen
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.9

# Feature-Liste
FEATURE_COLUMNS = [
    TARGET_COL,
    'Temp_Lag1', 'Temp_Lag2', 'Temp_Lag3',
    'Temp_Rolling_Mean_6', 'Temp_Rolling_Mean_12', 'Temp_Rolling_Mean_24',
    'Temp_Rolling_Std',
    'Hour', 'Month', 'DayOfWeek', 'Season',
    'IsWeekend'
]

N_FEATURES = len(FEATURE_COLUMNS)

def enhanced_feature_engineering(df):
    """Erstellt fortgeschrittene Features für das Modell."""
    df = df.copy()
    
    # Lags (Verzögerungswerte)
    for lag in [1, 2, 3]:
        df[f'Temp_Lag{lag}'] = df[TARGET_COL].shift(lag)
    
    # Rolling Windows (Gleitende Mittelwerte)
    for window in [6, 12, 24]:
        df[f'Temp_Rolling_Mean_{window}'] = df[TARGET_COL].rolling(window=window).mean()
    
    # Volatilität (Standardabweichung über 24h)
    df['Temp_Rolling_Std'] = df[TARGET_COL].rolling(window=24).std()
    
    # Zeitbasierte Features
    df['Hour'] = df.index.hour
    df['Month'] = df.index.month
    df['DayOfWeek'] = df.index.dayofweek
    df['Season'] = df.index.month % 12 // 3
    df['IsWeekend'] = (df.index.dayofweek >= 5).astype(int)
    
    return df

def create_sequences(data, target_col_idx, time_steps, forecast_horizon):
    """
    Erstellt Sequenzen (X) und Targets (y).
    Hier wird das Fenster über das Array geschoben.
    """
    X, y = [], []
    for i in range(time_steps, len(data) - forecast_horizon + 1):
        # Das Fenster reicht von i-72 bis i
        X.append(data[i-time_steps:i])
        # Das Target ist der Wert bei i + horizon - 1
        y.append(data[i + forecast_horizon - 1, target_col_idx])
    return np.array(X), np.array(y)

def prepare_lstm_data():
    """
    Optimierte Hauptfunktion: 
    Lädt Daten, erstellt Features, skaliert korrekt (kein Leakage) 
    und splittet Sequenzen ohne Lücken.
    """
    # 1. Daten laden
    df_raw = load_and_preprocess_data()
    if df_raw is None or df_raw.empty:
        return (None,) * 7

    # 2. Feature Engineering
    df_eng = enhanced_feature_engineering(df_raw)
    
    # Filtern der Spalten und Entfernen der NaNs (durch Lags/Rolling entstanden)
    df_eng = df_eng[FEATURE_COLUMNS].dropna()
    
    # 3. Bestimmung der Split-Indizes
    n = len(df_eng)
    train_idx_end = int(n * TRAIN_SPLIT)
    val_idx_end = int(n * VAL_SPLIT)
    
    # 4. Skalierung (WICHTIG: Fit nur auf Trainings-Datenbereich)
    scaler = MinMaxScaler()
    # Wir fitten den Scaler nur auf den Daten, die später zum Training gehören
    scaler.fit(df_eng.iloc[:train_idx_end])
    
    # Transformieren des gesamten Datensatzes
    data_scaled = scaler.transform(df_eng)
    
    # 5. Sequenzerstellung über das gesamte skalierte Array
    target_idx = FEATURE_COLUMNS.index(TARGET_COL)
    X_all, y_all = create_sequences(data_scaled, target_idx, TIME_STEPS, FORECAST_HORIZON)
    
    # 6. Aufteilung der fertigen Sequenzen
    # Da create_sequences erst bei Index 'TIME_STEPS' startet, 
    # müssen wir die Split-Punkte anpassen.
    split_train = train_idx_end - TIME_STEPS
    split_val = val_idx_end - TIME_STEPS
    
    X_train, y_train = X_all[:split_train], y_all[:split_train]
    X_val, y_val     = X_all[split_train:split_val], y_all[split_train:split_val]
    X_test, y_test   = X_all[split_val:], y_all[split_val:]
    
    print(f"--- Datenvorbereitung abgeschlossen ---")
    print(f"Features: {N_FEATURES} | Sequenzlänge: {TIME_STEPS}")
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

def get_feature_dataframe():
    """
    Loads raw data, applies feature engineering, and returns the complete feature DataFrame.
    Used for visualization purposes.
    """
    print("Loading data for feature visualization...")
    
    # Load raw data
    df_raw = load_and_preprocess_data()
    if df_raw is None or df_raw.empty:
        print("ERROR: Could not load raw data")
        return pd.DataFrame()
    
    # Apply feature engineering
    df_features = enhanced_feature_engineering(df_raw)
    
    # Filter to only include the feature columns we're using
    available_cols = [col for col in FEATURE_COLUMNS if col in df_features.columns]
    df_features = df_features[available_cols].dropna()
    
    print(f"Feature DataFrame created: {df_features.shape}")
    return df_features

if __name__ == '__main__':
    # Test-Lauf
    X_tr, y_tr, X_va, y_va, X_te, y_te, sc = prepare_lstm_data()