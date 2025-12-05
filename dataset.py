import requests
import pandas as pd
import time
import json
from pathlib import Path
import re

TEMP_FILE = "temp.json"
BLOCK_PATTERN = "usd_data_block_*.json"

def get_last_processed_date():
    """Oxirgi saqlangan sanani aniqlash."""
    files = sorted(Path('.').glob("usd_data_block_*.json"), key=lambda x: int(re.findall(r'\d+', x.stem)[0]))
    if files:
        last_file = files[-1]
        df = pd.read_json(last_file)
        return df['Date'].max(), len(files)
    elif Path(TEMP_FILE).exists():
        with open(TEMP_FILE, "r") as f:
            temp_data = json.load(f)
            return pd.to_datetime(temp_data.get("last_date")), 0
    else:
        return None, 0

def fetch_currency_archive(start_date="2018-12-01", end_date="2025-12-05", block_size=10, wait_sec=2):
    df_list = []
    
    last_date, counter = get_last_processed_date()
    if last_date is not None:
        current_date = pd.to_datetime(last_date) + pd.Timedelta(days=1)
    else:
        current_date = pd.to_datetime(start_date)
    
    end_date = pd.to_datetime(end_date)

    while current_date <= end_date:
        df_block = []
        for _ in range(block_size):
            if current_date > end_date:
                break
            url = f"https://cbu.uz/uz/arkhiv-kursov-valyut/json/all/{current_date.strftime('%Y-%m-%d')}/"
            try:
                response = requests.get(url)
                data = response.json()
            except Exception as e:
                print(f"{current_date.date()} da xatolik: {e}")
                current_date += pd.Timedelta(days=1)
                continue

            usd_data = [item for item in data if item.get('Ccy') == 'USD']
            if usd_data:
                df_block.append(pd.DataFrame(usd_data))

            current_date += pd.Timedelta(days=1)
        
        if df_block:
            df_block_concat = pd.concat(df_block, ignore_index=True)
            df_block_concat['Date'] = pd.to_datetime(df_block_concat['Date'], dayfirst=True)
            df_block_concat['Rate'] = df_block_concat['Rate'].astype(float)
            df_block_concat.sort_values('Date', inplace=True)
            df_block_concat.reset_index(drop=True, inplace=True)

            # Har blokni alohida faylga saqlash
            counter += 1
            file_name = f"usd_data_block_{counter}.json"
            df_block_concat.to_json(file_name, orient="records", date_format="iso")
            print(f"Blok {counter} saqlandi: {file_name}, kunlar: {df_block_concat['Date'].min().date()} - {df_block_concat['Date'].max().date()}")

            # Oxirgi sanani temp faylga yozish
            with open(TEMP_FILE, "w") as f:
                json.dump({"last_date": df_block_concat['Date'].max().strftime("%Y-%m-%d")}, f)

            time.sleep(wait_sec)
            df_list.append(df_block_concat)
    
    if df_list:
        full_df = pd.concat(df_list, ignore_index=True)
        full_df.sort_values('Date', inplace=True)
        full_df.reset_index(drop=True, inplace=True)
        full_df.to_json("usd_data_full.json", orient="records", date_format="iso")
        return full_df
    else:
        raise ValueError("USD ma'lumotlari topilmadi")

# Foydalanish:
df = fetch_currency_archive()
print("To'liq dataset yuklandi:", df.shape)
