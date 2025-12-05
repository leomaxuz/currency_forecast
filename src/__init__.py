from src.fetch_data import fetch_currency_data
from src.preprocess import create_features
from src.model import train_model
from src.visualize import plot_forecast

# 1. Ma'lumot olish
df = fetch_currency_data()

# 2. Feature yaratish
X, y = create_features(df, window=7)

# 3. Model yaratish
model, scaler = train_model(X, y)

# 4. Bashorat qilish
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

# 5. Grafik chizish
plot_forecast(df, y_pred)
