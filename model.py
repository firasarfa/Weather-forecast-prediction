import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers 
from lstm_preparation import prepare_lstm_data, TIME_STEPS, N_FEATURES 

# --- KONFIGURATION ---
# Erhöhte Kapazität für das multivariate Modell
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DROPOUT_RATE = 0.3
L2_REG_STRENGTH = 0.00005
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10

def build_improved_lstm_model(time_steps, n_features):
    """
    Erstellt ein verbessertes LSTM-Modell.
    Nutzt Huber-Loss für Robustheit gegenüber Ausreißern und L2-Regularisierung.
    """
    l2_reg = regularizers.L2(L2_REG_STRENGTH)
    
    model = Sequential([
        # Erste LSTM-Schicht mit 128 Units
        LSTM(units=LSTM_UNITS_1, return_sequences=True, 
             input_shape=(time_steps, n_features),
             kernel_regularizer=l2_reg),
        Dropout(DROPOUT_RATE),
        
        # Zweite LSTM-Schicht mit 64 Units
        LSTM(units=LSTM_UNITS_2, return_sequences=False,
             kernel_regularizer=l2_reg),
        Dropout(DROPOUT_RATE),
        
        # Zusätzliche Dense-Schicht zur tieferen Merkmalsextraktion
        Dense(units=16, activation='relu'),
        # Finale Vorhersageeinheit
        Dense(units=1)
    ])
    
    # Verwendung des Huber-Loss (Kombination aus MSE und MAE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
    
    model.summary()
    return model

def train_improved_model():
    """
    Lädt die optimierten Daten aus der Canvas-Vorbereitung und trainiert das Modell.
    """
    # Laden der 7 Rückgabewerte aus lstm_preparation.py
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_lstm_data()
    
    if X_train is None:
        print("Fehler beim Laden der Daten aus dem Vorbereitungs-Modul.")
        return None

    # Modell basierend auf den dynamischen Konstanten TIME_STEPS und N_FEATURES bauen
    model = build_improved_lstm_model(TIME_STEPS, N_FEATURES)
    
    # Definition der Callbacks für optimiertes Training
    callbacks = [
        # Stoppt, wenn val_loss sich 10 Epochen lang nicht verbessert
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        # Reduziert Lernrate bei Plateaus
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        # Speichert den besten Zustand
        ModelCheckpoint('best_rnn_model.keras', save_best_only=True)
    ]
    
    print("\n--- Starte verbessertes Modelltraining ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history, X_test, y_test, scaler

if __name__ == '__main__':
    # Startet das Training direkt
    train_improved_model()