# Currency Forecast (Valyuta Kursi Prognozi)

Ushbu loyiha O'zbekiston Markaziy Banki (CBU) ma'lumotlariga asoslanib, AQSh dollari (USD) kursini sun'iy intellekt (**LSTM**) yordamida prognoz qiladi.

## Xususiyatlari
*   **Avtomatik Ma'lumot Yig'ish**: Internetdan eng so'nggi kurslarni yuklab oladi (`dataset.py`).
*   **Retry Logic**: Internet uzilishlarida qayta urinish tizimi mavjud.
*   **Deep Learning (LSTM)**: An'anaviy usullardan ko'ra aniqroq bo'lgan neyron tarmoq modeli.
*   **Interaktiv Grafika**: `Plotly` yordamida harakatlanuvchi va ma'lumotga boy grafiklar.
*   **Ma'lumotlarni Saqlash**: Barcha ma'lumotlar `data/` papkasida tartibli saqlanadi.

## O'rnatish

1.  Ushbu repozitoriyni yuklab oling.
2.  Kerakli kutubxonalarni o'rnating:
    ```bash
    pip install -r requirements.txt
    ```

## Ishlatish

Dasturni ishga tushirish uchun:

```bash
python main.py
```

Bu buyruq quyidagilarni bajaradi:
1.  Yangi ma'lumotlarni tekshiradi va yuklaydi.
2.  Agar model yo'q bo'lsa, yangi LSTM modelini o'qitadi (bu biroz vaqt olishi mumkin).
3.  Keyingi 30 kunlik kursni prognoz qiladi.
4.  Brauzerda interaktiv grafikni ochadi.

## Fayl Tuzilmasi

*   `main.py` - Asosiy dastur (Model, Prognoz, Vizualizatsiya).
*   `dataset.py` - Ma'lumotlarni yig'ish va tozalash moduli.
*   `data/` - Ma'lumotlar va saqlangan modellar papkasi.
*   `requirements.txt` - Kerakli kutubxonalar ro'yxati.

## Muallif
Loyiha yangilandi va refactoring qilindi.
