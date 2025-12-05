# Currency Forecast

Valyuta kursini bashorat qilish uchun Python loyihasi. Ushbu loyiha API orqali ma'lumotlarni oladi, ularni tozalaydi, bashorat modelini yaratadi va natijalarni vizual tarzda ko‘rsatadi.

---

## Loyihaning tuzilishi

currency_forecast/

─ data/                  # API orqali olingan ma'lumotlar saqlanadi

─ notebooks/             # Jupyter notebook (agar ishlatilsa)

─ src/                   # Asosiy kodlar
   ├─ fetch_data.py      # API dan ma'lumot olish
   
   ├─ preprocess.py      # Ma'lumotni tozalash
   ├─ model.py           # Bashorat modelini yaratish
   ├─ visualize.py       # Grafik chizish
   └─ main.py            # Dastur ishga tushiriladi
─ requirements.txt       # Kutubxonalar
─ README.md              # Loyihaning tavsifi
─ .gitignore

---

## Kutubxonalarni o‘rnatish

pip install -r requirements.txt

---

## Dastur ishga tushirish

python src/main.py

---

## Foydalanish bosqichlari

1. fetch_data.py orqali API dan ma'lumot oling.
2. preprocess.py bilan ma'lumotlarni tozalang.
3. model.py yordamida bashorat modelini yaratib, uni main.py orqali ishga tushiring.
4. visualize.py bilan natijalarni grafik ko‘rinishda tahlil qiling.

---

## Litsenziya

MIT License
