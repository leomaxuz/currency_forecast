import matplotlib.pyplot as plt

def plot_forecast(df, y_pred, window=7):
    df_plot = df.copy()
    df_plot['Forecast'] = pd.Series(y_pred, index=df_plot.index[:len(y_pred)])
    
    plt.figure(figsize=(12,6))
    plt.plot(df_plot['Date'], df_plot['Rate'], label='Real')
    plt.plot(df_plot['Date'], df_plot['Forecast'], label='Forecast', linestyle='--')
    
    # Ko'tarilgan va tushgan kunlarni belgilash
    df_plot['Diff'] = df_plot['Forecast'] - df_plot['Rate']
    plt.scatter(df_plot['Date'][df_plot['Diff']>0], df_plot['Forecast'][df_plot['Diff']>0], color='green', label='Increase')
    plt.scatter(df_plot['Date'][df_plot['Diff']<0], df_plot['Forecast'][df_plot['Diff']<0], color='red', label='Decrease')
    
    plt.xlabel("Date")
    plt.ylabel("USD/UZS Rate")
    plt.legend()
    plt.show()
