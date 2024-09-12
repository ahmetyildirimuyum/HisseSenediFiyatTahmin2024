import os
import pandas as pd
import matplotlib.pyplot as plt

# Dosya yolu ve sayfa adı
file_path = 'D:/UyumProjects/UyumERP/trunk/duzelt/Ahmet_YILDIRIM/_YUKSEK_LISANS_TEZ/_Notlar/2024_09_11__18_00/2019_TAHMIN_DETAYLARI.xlsx'
sheet_name = 'Tüm Tahminler'

# Excel dosyasını okuma
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Grafiklerin kaydedileceği dizin
output_dir = "D:/UyumProjects/UyumERP/trunk/duzelt/Ahmet_YILDIRIM/_YUKSEK_LISANS_TEZ/_Notlar/2024_09_11__18_00_Sonuc_Grafik"

# Eğer dizin yoksa oluştur
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Hisse senedi isimlerini alma
stocks = df['Hisse Senedi'].unique()

# Hisse senedi bazında ilk set grafik (1. Grafik)
for stock in stocks:
    stock_data = df[df['Hisse Senedi'] == stock]

    # 1. Grafik: Gerçek değer ve tahminler
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Tarih'], stock_data['Gerçek Değer'], label='Gerçek Değer', color='blue')
    plt.plot(stock_data['Tarih'], stock_data['ARIMA'], label='ARIMA', color='green')
    plt.plot(stock_data['Tarih'], stock_data['XGBoost'], label='XGBoost', color='red')
    plt.plot(stock_data['Tarih'], stock_data['Prophet'], label='Prophet', color='purple')
    plt.plot(stock_data['Tarih'], stock_data['LSTM'], label='LSTM', color='orange')

    plt.title(f"{stock} Tahmin Karşılaştırması - Gerçek Değer ve Tahminler")
    plt.xlabel("Tarih")
    plt.ylabel("Değer")
    plt.legend()
    
    # 1. Grafik için dosya adı ve kaydetme işlemi
    output_file_1 = os.path.join(output_dir, f"{stock}_tahmin_karsilastirma.png")
    plt.savefig(output_file_1)
    plt.close()

# Hisse senedi bazında ikinci set grafik (2. Grafik)
for stock in stocks:
    stock_data = df[df['Hisse Senedi'] == stock]

    # 2. Grafik: Doğruluk karşılaştırmaları
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Tarih'], stock_data['ARIMA Doğruluk'], label='ARIMA Doğruluk', color='green')
    plt.plot(stock_data['Tarih'], stock_data['XGBoost Doğruluk'], label='XGBoost Doğruluk', color='red')
    plt.plot(stock_data['Tarih'], stock_data['Prophet Doğruluk'], label='Prophet Doğruluk', color='purple')
    plt.plot(stock_data['Tarih'], stock_data['LSTM Doğruluk'], label='LSTM Doğruluk', color='orange')

    plt.title(f"{stock} Tahmin Doğruluk Karşılaştırması")
    plt.xlabel("Tarih")
    plt.ylabel("Doğruluk")
    plt.legend()

    # 2. Grafik için dosya adı ve kaydetme işlemi
    output_file_2 = os.path.join(output_dir, f"{stock}_dogruluk_karsilastirma.png")
    plt.savefig(output_file_2)
    plt.close()
