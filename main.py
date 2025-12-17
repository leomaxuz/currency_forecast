"""
main.py

Valyuta kursini bashorat qilish (USD/UZS).
LSTM (Long Short-Term Memory) Deep Learning modeli va Plotly vizualizatsiyasi.
"""

import os
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Local imports
from dataset import update_dataset, DATA_DIR

# Loglash
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Konfiguratsiya
MODEL_FILE = DATA_DIR / "lstm_model.h5"
SCALER_FILE = DATA_DIR / "scaler.pkl"
LOOK_BACK = 60  # O'tmishdagi necha kunga qarab bashorat qilish

def prepare_data_for_lstm(data, look_back=60):
    """
    LSTM uchun ma'lumotlarni tayyorlash.
    X: (samples, look_back, 1)
    y: (samples, 1)
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """LSTM model arxitekturasini qurish."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_save_model(df):
    """Modelni o'qitish va saqlash."""
    data = df['Rate'].values.reshape(-1, 1)
    
    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Train/Test split (80/20)
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - LOOK_BACK:]
    
    X_train, y_train = prepare_data_for_lstm(train_data, LOOK_BACK)
    X_test, y_test = prepare_data_for_lstm(test_data, LOOK_BACK)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    logger.info(f"Model o'qitilmoqda... (Train: {X_train.shape[0]}, Test: {X_test.shape[0]})")
    
    model = build_lstm_model((X_train.shape[1], 1))
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train, 
        epochs=50, 
        batch_size=32, 
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Saqlash
    model.save(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    logger.info("Model va Scaler saqlandi.")
    
    # Baholash
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mae = mean_absolute_error(y_test_real, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_real, predictions))
    logger.info(f"Model Xatoligi: MAE={mae:.2f}, RMSE={rmse:.2f}")
    
    return model, scaler

def load_model_and_scaler():
    """Saqlangan modelni yuklash."""
    if MODEL_FILE.exists() and SCALER_FILE.exists():
        model = tf.keras.models.load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler
    return None, None

def predict_future(model, scaler, df, days=7):
    """Kelajakni bashorat qilish."""
    last_data = df['Rate'].values[-LOOK_BACK:].reshape(-1, 1)
    current_batch = scaler.transform(last_data)
    current_batch = current_batch.reshape(1, LOOK_BACK, 1)
    
    future_predictions = []
    
    for _ in range(days):
        pred = model.predict(current_batch)[0]
        future_predictions.append(pred[0])
        
        # Yangi bashoratni qo'shib, eskini chiqarib tashlaymiz
        current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
        
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    last_date = df['Date'].max()
    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days)]
    
    return pd.DataFrame({'Date': future_dates, 'Rate': future_predictions.flatten()})

def plot_interactive(df, forecast_df):
    """Plotly yordamida interaktiv grafik chizish."""
    fig = go.Figure()

    # Tarixiy ma'lumotlar
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Rate'],
        mode='lines', name='Tarixiy Kurs',
        line=dict(color='royalblue', width=2)
    ))

    # Prognoz
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'], y=forecast_df['Rate'],
        mode='lines+markers', name='Prognoz',
        line=dict(color='firebrick', width=3, dash='dot')
    ))

    fig.update_layout(
        title='USD/UZS Valyuta Kursi Prognozi (LSTM)',
        xaxis_title='Sana',
        yaxis_title='Kurs (SO\'M)',
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    fig.show()

def main():
    # 1. Ma'lumotlarni yangilash
    df = update_dataset()
    if df.empty:
        logger.error("Ma'lumotlar yo'q. Dastur to'xtatildi.")
        return

    # 2. Modelni yuklash yoki o'qitish
    model, scaler = load_model_and_scaler()
    if model is None:
        logger.info("Yangi model o'qitilmoqda...")
        model, scaler = train_and_save_model(df)
    else:
        logger.info("Mavjud model yuklandi.")
        
        # Agar yangi ma'lumotlar ko'p bo'lsa, qayta o'qitish kerak bo'lishi mumkin
        # Hozircha oddiylik uchun har doim eskisini ishlatamiz yoki flag bilan boshqarish mumkin
        # Keling, foydalanuvchi so'rasa qayta o'qitamiz (force_train=False)

    # 3. Prognoz qilish
    days = 30
    logger.info(f"Keyingi {days} kun uchun prognoz qilinmoqda...")
    forecast_df = predict_future(model, scaler, df, days)
    
    print("\n--- PROGNOZ ---")
    print(forecast_df)
    
    # 4. Vizualizatsiya
    plot_interactive(df.tail(365), forecast_df) # Grafikda faqat oxirgi 1 yilni ko'rsatamiz

if __name__ == "__main__":
    main()
