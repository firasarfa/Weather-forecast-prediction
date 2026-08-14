# Weather Forecast Prediction - LSTM Model

## Overview
This is a project from my university Machine Learning module. It uses a multivariate LSTM neural network to forecast hourly temperatures based on historical weather data from 2014 to 2016.

## Tech Stack
- **Python 3.x**
- **TensorFlow 2.x** & Keras - LSTM model architecture and training
- **pandas** & **numpy** - Data preprocessing and feature engineering
- **scikit-learn** - Min-Max scaling and train/val/test splitting
- **matplotlib** & **seaborn** - Data visualization and diagnostics

## What It Does
- Loads and preprocesses 6 CSV weather datasets (2014-2016, hourly resolution)
- Engineers 13 advanced features including lag values, rolling statistics (6/12/24h), seasonal markers, and time-of-day flags
- Scales data with MinMaxScaler (fit only on training data to prevent leakage)
- Trains an improved LSTM model with:
  - 2-layer LSTM architecture (128 → 64 units)
  - L2 regularization
  - Dropout (0.3)
  - Huber loss function for robustness to outliers
  - Early stopping and learning rate reduction callbacks
- Evaluates model performance with metrics (MSE, MAE, R²) and compares against a naive persistence baseline
- Generates comprehensive visualizations (time series, actual vs. predicted, residuals distribution, error over time)

## Project Structure
```
weather-forcast/
├── model.py              # LSTM model architecture and training logic
├── lstm_preparation.py   # Feature engineering, sequence creation, data scaling
├── data_prep.py          # Data loading, cleaning, resampling
├── model_evaluation.py   # Model evaluation, metrics, and visualization
├── data_visualization.py # Exploratory data analysis plots
├── .env.example        # Environment variable templates
├── .gitignore          # OS/IDE specific ignore rules
└── *.csv               # Weather data datasets (mpi_roof_2014a-b, 2015a-b, 2016a-b)
```

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/weather-forecast-prediction.git
cd weather-forecast-prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
# Core dependencies: tensorflow, pandas, numpy, scikit-learn, matplotlib, seaborn
```

### 4. Train the model
```bash
python model.py
```
This will train the LSTM model and save `best_rnn_model.keras` using ModelCheckpoint.

### 5. Evaluate the model
```bash
python model_evaluation.py
```
Loads the trained model and generates performance reports + visualization plots.

### 6. Run data preprocessing
```bash
python data_prep.py
```

### 7. Run feature engineering and LSTM data preparation
```bash
python lstm_preparation.py
```

### 8. Run data visualization
```bash
python data_visualization.py
```

## Results
- **Test MAE**: Mean absolute error in °C (varies by data split)
- **Test R²**: Coefficient of determination
- **Baseline comparison**: Model performance is compared against a persistence (naive) baseline
- **Visualizations**: Generated plots show time-series fit, actual vs. predicted scatter, residual distribution, and error over time

## Notes
- The scaler is fit only on training data to prevent data leakage
- Training uses EarlyStopping (patience=10) and ModelCheckpoint to save the best model
- All CSV data files are included in the repo; ensure they remain in the project root
- For production use, consider adding a `config.py` or environment variables for hyperparameters