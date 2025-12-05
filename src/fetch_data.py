import requests
import pandas as pd

def fetch_currency_data(start_date="2018-12-01"):
    url = f"https://cbu.uz/uz/arkhiv-kursov-valyut/json/all/{start_date}/"
    response = requests.get(url)
    data = response.json()
    
    # Faqat USD kursi
    usd_data = [item for item in data if item['Ccy'] == 'USD']
    
    df = pd.DataFrame(usd_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Rate'] = df['Rate'].astype(float)
    df.sort_values('Date', inplace=True)
    return df[['Date', 'Rate']]
