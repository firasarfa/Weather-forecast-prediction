import pandas as pd
import numpy as np

# --- Konfiguration ---
# Dies sind die Konstanten, die das Modul benötigt
TEMP_COL = 'T (degC)'
CSV_FILES = [
    'mpi_roof_2014a.csv', 'mpi_roof_2014b.csv', 'mpi_roof_2015a.csv', 
    'mpi_roof_2015b.csv', 'mpi_roof_2016a.csv', 'mpi_roof_2016b.csv'
]
START_DATE = '2014-01-01'
END_DATE = '2016-12-31'
RESAMPLE_FREQ = '1H' # Reduktion auf stündliche Mittelwerte (Haupt-Frequenz für LSTM)

def load_and_preprocess_data():
    """
    Lädt alle CSV-Dateien, führt sie zusammen, bereinigt den DatetimeIndex,
    filtert den Zeitraum (2014-2016), führt Downsampling (1H) durch, behandelt
    NaN-Werte und gibt das finale, univariate Dataframe mit dem Temperatur-Feature
    (T (degC)) zurück.
    
    Rückgabe: pandas.DataFrame (indexiert nach Zeit, nur eine Spalte: T (degC))
    """
    print("--- Datenprozessor gestartet ---")
    
    # 1. Laden und Zusammenfügen der Daten
    dataframes = []
    for file in CSV_FILES:
        try:
            df_temp = pd.read_csv(file, encoding='latin-1')
            dataframes.append(df_temp)
        except FileNotFoundError:
            continue
    
    if not dataframes:
        print("FEHLER: Konnte keine der Quelldateien laden.")
        return pd.DataFrame()

    df = pd.concat(dataframes, ignore_index=True)
    print(f"Alle Dateien erfolgreich zusammengeführt. Gesamtform: {df.shape}")

    # 2. Indexierung und Bereinigung
    # Fehlerhafte Datumsformate werden zu NaT
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    
    # Entferne Zeilen, bei denen die 'Date Time' Konvertierung fehlgeschlagen ist (NaT)
    df = df.dropna(subset=['Date Time'])
    
    # Setze 'Date Time' als Index
    df.set_index('Date Time', inplace=True)
    
    # Bereinigung: Duplikate und Sortierung
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    # 3. Filtern und Downsampling
    df_filtered = df.loc[START_DATE:END_DATE].copy()
    if df_filtered.empty:
        print(f"FEHLER: Nach dem Filtern auf {START_DATE}-{END_DATE} ist der DataFrame leer.")
        return pd.DataFrame()

    # Reduktion auf stündliche Mittelwerte (Primary Data for LSTM)
    df_hourly = df_filtered.resample(RESAMPLE_FREQ).mean()
    print(f"Daten auf stündliche Mittelwerte ({RESAMPLE_FREQ}) reduziert. Finale Form: {df_hourly.shape}")
    
    # NEU: Aggregation auf tägliche Mittelwerte (Zusätzliche Analyse)
    df_daily = df_hourly.resample('D').mean()
    print(f"Zusätzliche Aggregation: Tägliche Mittelwerte (Frequenz 'D') berechnet. Form: {df_daily.shape}")
    
    # 4. Fehlwertbehandlung (ffill) für stündliche Daten
    nan_counts_hourly = df_hourly.isnull().sum().sum()
    if nan_counts_hourly > 0:
        # Füllen der Lücken und Entfernen eventuell verbleibender NaNs am Anfang
        # Anwendung auf df_hourly, da dies das Haupt-DF ist.
        df_hourly.fillna(method='ffill', inplace=True)
        df_hourly.dropna(inplace=True)
        print(f"WARNUNG: {nan_counts_hourly} NaN-Werte in stündlichen Daten wurden mit 'ffill' ersetzt/entfernt.")

    # 5. Rückgabe des univariaten Datensatzes (nur Temperatur)
    df_final = df_hourly[[TEMP_COL]]
    print(f"Datenprozessor erfolgreich abgeschlossen. Finale (stündliche) Datenpunkte: {df_final.shape[0]}")
    return df_final

if __name__ == '__main__':
    # Beispielaufruf, um die Funktionalität des Moduls zu testen
    final_data = load_and_preprocess_data()
    if not final_data.empty:
        print("\nHead des finalen (stündlichen) Dataframes:")
        print(final_data.head())
        print(f"\nDatumsbereich: {final_data.index.min()} bis {final_data.index.max()}") 
