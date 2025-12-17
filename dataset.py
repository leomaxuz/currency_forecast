import requests
import pandas as pd
import time
import json
from pathlib import Path
import logging
from datetime import datetime, timedelta
import random

# Loglashni sozlash
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Konfiguratsiya
DATA_DIR = Path("data")
TEMP_FILE = DATA_DIR / "temp_last_date.json"
FULL_DATA_FILE = DATA_DIR / "usd_data_full.json"
BASE_URL = "https://cbu.uz/uz/arkhiv-kursov-valyut/json/all/{date}/"

def ensure_data_dir():
    """Ma'lumotlar papkasini yaratish."""
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_last_processed_date(start_date: str) -> str:
    """
    Oxirgi saqlangan sanani aniqlash. 
    Agar fayllar bo'lmasa, start_date qaytariladi.
    """
    ensure_data_dir()
    
    # 1. Avval temp faylni tekshiramiz
    if TEMP_FILE.exists():
        try:
            with open(TEMP_FILE, "r") as f:
                data = json.load(f)
                return data.get("last_date", start_date)
        except Exception:
            pass
            
    # 2. Agar temp bo'lmasa, mavjud fayllardan oxirgisini qidiramiz
    files = sorted(DATA_DIR.glob("usd_data_block_*.json"), key=lambda x: int(x.stem.split('_')[-1]))
    if files:
        try:
            last_file = files[-1]
            df = pd.read_json(last_file)
            if not df.empty and 'Date' in df.columns:
                return pd.to_datetime(df['Date']).max().strftime("%Y-%m-%d")
        except Exception as e:
            logger.error(f"Faylni o'qishda xatolik: {e}")
            
    return start_date

def fetch_data_with_retry(url: str, max_retries: int = 3) -> list:
    """URL dan ma'lumotni retry (qayta urinish) bilan olish."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"Urinish {attempt + 1}/{max_retries} xato berdi: {e}")
            time.sleep(2 + random.random()) # Biroz kutish
    logger.error(f"Ma'lumot olinmadi: {url}")
    return []

def update_dataset(start_date: str = "2018-01-01", end_date: str = None, block_size: int = 10):
    """
    Datasetni yangilash funksiyasi.
    """
    ensure_data_dir()
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    last_date_str = get_last_processed_date(start_date)
    current_date = pd.to_datetime(last_date_str) + timedelta(days=1)
    # Agar start_date va topilgan sana bir xil bo'lsa va biz boshidan boshlamasak, demak 1 kun qo'shish kerak emas.
    # Lekin mantiqan: agar last_date "2018-01-01" bo'lsa, demak bu kun bormi?
    # Keling, oddiyroq qilamiz: agar fayllar bo'lmasa, current_date = start_date.
    if not list(DATA_DIR.glob("usd_data_block_*.json")) and not TEMP_FILE.exists():
        current_date = pd.to_datetime(start_date)
    
    end_date_dt = pd.to_datetime(end_date)

    if current_date > end_date_dt:
        logger.info("Ma'lumotlar allaqachon yangilangan.")
        return load_full_data()

    logger.info(f"Ma'lumotlarni yuklash boshlandi: {current_date.date()} dan {end_date_dt.date()} gacha")
    
    df_list = []
    
    # Mavjud blok fayllarni sanash
    existing_blocks = len(list(DATA_DIR.glob("usd_data_block_*.json")))
    block_counter = existing_blocks

    while current_date <= end_date_dt:
        block_data = []
        for _ in range(block_size):
            if current_date > end_date_dt:
                break
            
            url = BASE_URL.format(date=current_date.strftime('%Y-%m-%d'))
            data = fetch_data_with_retry(url)
            
            usd_data = [item for item in data if item.get('Ccy') == 'USD']
            if usd_data:
                # Sanani ham qo'shamiz, chunki javob ichida ba'zan sana bo'lmasligi mumkin
                item = usd_data[0]
                item['Date'] = current_date.strftime('%Y-%m-%d')
                block_data.append(item)
            
            current_date += timedelta(days=1)
            time.sleep(0.1) # Serverni juda yuklamaslik uchun

        if block_data:
            block_counter += 1
            df_block = pd.DataFrame(block_data)
            
            # Tiplarni to'g'irlash
            df_block['Rate'] = df_block['Rate'].astype(float)
            df_block['Date'] = pd.to_datetime(df_block['Date'])
            
            # Blokni saqlash
            block_filename = DATA_DIR / f"usd_data_block_{block_counter}.json"
            df_block.to_json(block_filename, orient="records", date_format="iso")
            logger.info(f"Blok {block_counter} saqlandi ({len(df_block)} kun).")
            
            # Temp faylni yangilash
            with open(TEMP_FILE, "w") as f:
                json.dump({"last_date": df_block['Date'].max().strftime("%Y-%m-%d")}, f)

    # Barcha bloklarni birlashtirib, katta fayl hosil qilish
    return compile_full_dataset()

def compile_full_dataset():
    """Barcha blok fayllarni bitta json ga yig'ish."""
    ensure_data_dir()
    files = list(DATA_DIR.glob("usd_data_block_*.json"))
    if not files:
        logger.warning("Hali hech qanday ma'lumot yuklanmagan.")
        return pd.DataFrame()

    df_list = [pd.read_json(f) for f in files]
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Tozalash va tartiblash
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    full_df.sort_values('Date', inplace=True)
    full_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
    full_df.reset_index(drop=True, inplace=True)
    
    full_df.to_json(FULL_DATA_FILE, orient="records", date_format="iso")
    logger.info(f"To'liq dataset saqlandi: {FULL_DATA_FILE} ({len(full_df)} qator)")
    return full_df

def load_full_data():
    """Tayyor datasetni o'qish."""
    if FULL_DATA_FILE.exists():
        df = pd.read_json(FULL_DATA_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    else:
        return compile_full_dataset()

if __name__ == "__main__":
    # Test uchun ishga tushirish
    df = update_dataset()
    print(df.tail())
