import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def train_model(X, y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_scaled, y, epochs=50, batch_size=16, verbose=1)
    return model, scaler
