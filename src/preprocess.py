def create_features(df, window=7):
    df = df.copy()
    df['Target'] = df['Rate'].shift(-window)
    df.dropna(inplace=True)
    X = df['Rate'].values.reshape(-1, 1)
    y = df['Target'].values
    return X, y
