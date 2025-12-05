"""
main.py

Valyuta kursini bashorat qilish dasturi.
O'z Res Markaziy bankining API ma'lumotlariga asoslanadi.
"""

import os
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt

# =====================
# 1. Ma'lumotlarni olish / JSON’dan yuklash
# =====================
def fetch_currency_archive(start_date="2018-12-01", end_date="2025-12-05", json_file="usd_data_full.json"):
    # Agar JSON mavjud bo'lsa, uni o'qish
    if os.path.exists(json_file):
        df = pd.read_json(json_file)
    else:
        df_list = []
        current_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        while current_date <= end_date:
            url = f"https://cbu.uz/uz/arkhiv-kursov-valyut/json/all/{current_date.strftime('%Y-%m-%d')}/"
            response = requests.get(url)
            data = response.json()

            usd_data = [item for item in data if item.get('Ccy') == 'USD']
            if usd_data:
                df_list.append(pd.DataFrame(usd_data))

            current_date += pd.Timedelta(days=1)

        if not df_list:
            raise ValueError("USD ma'lumotlari topilmadi")

        df = pd.concat(df_list, ignore_index=True)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df['Rate'] = df['Rate'].astype(float)
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_json(json_file, orient="records", date_format="iso")
    return df[['Date', 'Rate']]

# =====================
# 2. Modelni tayyorlash
# =====================
def train_model(X, y, model_file="usd_model.pkl"):
    if os.path.exists(model_file):
        # Model saqlangan bo'lsa, uni yuklash
        model, scaler = joblib.load(model_file)
    else:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)
        joblib.dump((model, scaler), model_file)
    return model, scaler

# =====================
# 3. Prognoz qilish
# =====================
def forecast(df, days=7):
    X = df[['Rate']]
    y = df['Rate'].shift(-1)[:-1]
    X = X[:-1]

    if X.empty or y.empty:
        raise ValueError("X yoki y bo'sh, modelni o'qitib bo'lmaydi")

    model, scaler = train_model(X, y)
    last_rate = X.iloc[-1].values.reshape(1, -1)
    forecasted = []

    for _ in range(days):
        scaled = scaler.transform(last_rate)
        pred = model.predict(scaled)[0]
        forecasted.append(pred)
        last_rate = np.array([[pred]])

    forecast_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Rate': forecasted})
    return forecast_df

# =====================
# 4. Grafik chizish
# =====================
def plot_forecast(df, forecast_df):
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Rate'], label='Real Rate', color='blue')
    plt.plot(forecast_df['Date'], forecast_df['Rate'], label='Forecast Rate', color='orange')

    up_days = df[df['Rate'].diff() > 0]
    down_days = df[df['Rate'].diff() < 0]

    plt.scatter(up_days['Date'], up_days['Rate'], color='green', label='Up Day', marker='^')
    plt.scatter(down_days['Date'], down_days['Rate'], color='red', label='Down Day', marker='v')

    plt.title("USD kursi (Real va Prognoz)")
    plt.xlabel("Sana")
    plt.ylabel("Rate (UZS)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =====================
# 5. Ishga tushirish
# =====================
if __name__ == "__main__":
    try:
        df = fetch_currency_archive()
        print("Olingan ma'lumotlar:", df.shape)

        forecast_7 = forecast(df, days=7)
        forecast_30 = forecast(df, days=30)

        print("So'nggi 7 kunlik prognoz:")
        print(forecast_7)

        print("So'nggi 30 kunlik prognoz:")
        print(forecast_30)

        plot_forecast(df, forecast_30)

    except Exception as e:
        print("Xatolik yuz berdi:", e)
