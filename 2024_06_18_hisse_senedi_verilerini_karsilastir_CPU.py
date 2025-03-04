# Kullanılan algoritmalar
# 1. ARIMA
# 2. XGBoost
# 3. Prophet
# 4. LSTM

import os
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
from tbats import TBATS
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.deterministic import Fourier
from pykalman import KalmanFilter
import warnings

# Uyarıları yok sayma
warnings.filterwarnings("ignore", message="ConvergenceWarning")

# Veri İndirme
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]  # Örnek hisseler
start_date = "2010-01-01"
end_date = "2019-12-31"
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# NaN değerleri doldurma veya kaldırma
data = data.ffill().bfill()

# Tarih indeksine frekans ekleme
data.index = pd.to_datetime(data.index)
data = data.asfreq('B')

# Tekrar NaN değer kontrolü ve doldurma
data = data.ffill().bfill()
data = data.dropna()

# Eğitim ve Test Verileri (normalize edilmemiş)
train_raw = data[:'2018']
test_raw = data['2019-01-01':'2019-12-31']

# NaN değerlerin olmadığını kontrol etme
assert train_raw.isna().sum().sum() == 0, "Train data contains NaN values"
assert test_raw.isna().sum().sum() == 0, "Test data contains NaN values"

# MinMaxScaler'ı tanımlayarak feature_range parametresini (-1, 1) olarak ayarlayın
scaler = MinMaxScaler(feature_range=(-1, 1))

# Verileri normalize edin ve sonuçları bir DataFrame'e dönüştürün
data_normalized = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)

# Eğitim ve Test Verileri (normalize edilmiş)
train = data_normalized[:'2018']
test = data_normalized['2019-01-01':'2019-12-31']

# NaN değerlerin olmadığını kontrol etme
assert train.isna().sum().sum() == 0, "Train data contains NaN values"
assert test.isna().sum().sum() == 0, "Test data contains NaN values"

# Hata metriklerini hesaplamak için fonksiyon
def calculate_errors(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, rmse, mae

# Dosya oluşturma işlemi
today = datetime.today().strftime('%Y_%m_%d__%H_%M')
output_dir = f"D:\\UyumProjects\\UyumERP\\trunk\\duzelt\\Ahmet_YILDIRIM\\_YUKSEK_LISANS_TEZ\\_Notlar\\{today}"
os.makedirs(output_dir, exist_ok=True)

# Eğitim ve test verilerini kaydetme (normalize edilmemiş)
train_raw.to_csv(os.path.join(output_dir, "train_data_raw.csv"))
test_raw.to_csv(os.path.join(output_dir, "test_data_raw.csv"))

# Eğitim ve test verilerini kaydetme (normalize edilmiş)
train.to_csv(os.path.join(output_dir, "train_data_normalized.csv"))
test.to_csv(os.path.join(output_dir, "test_data_normalized.csv"))

# Sonuçları kaydetmek için veri çerçevesi
results = pd.DataFrame(columns=["Ticker", "Algorithm", "MSE", "RMSE", "MAE"])

# Algoritmaları ve hisseleri döngüye alarak tahminlerin hatalarını hesaplamak
for ticker in tickers:
    y_train = train[ticker]
    y_test = test[ticker]
    
    # ARIMA (Örnek olarak basit bir ARIMA modeli)
    model_arima = ARIMA(y_train, order=(5,1,0)).fit()
    y_pred_arima = model_arima.forecast(steps=len(y_test))
    
    mse_arima, rmse_arima, mae_arima = calculate_errors(y_test, y_pred_arima)
    new_result = pd.DataFrame({"Ticker": [ticker], "Algorithm": ["ARIMA"], "MSE": [mse_arima], "RMSE": [rmse_arima], "MAE": [mae_arima]})
    if not new_result.empty:
        results = pd.concat([results, new_result.dropna(how='all')], ignore_index=True)
    
    # XGBoost
    model_xgb = xgb.XGBRegressor(objective='reg:squarederror')
    model_xgb.fit(np.arange(len(y_train)).reshape(-1, 1), y_train)
    y_pred_xgb = model_xgb.predict(np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1))
    
    mse_xgb, rmse_xgb, mae_xgb = calculate_errors(y_test, y_pred_xgb)
    new_result = pd.DataFrame({"Ticker": [ticker], "Algorithm": ["XGBoost"], "MSE": [mse_xgb], "RMSE": [rmse_xgb], "MAE": [mae_xgb]})
    if not new_result.empty:
        results = pd.concat([results, new_result.dropna(how='all')], ignore_index=True)
    
    # Prophet
    df_train = y_train.reset_index().rename(columns={'Date': 'ds', ticker: 'y'})
    model_prophet = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model_prophet.fit(df_train)
    
    future = model_prophet.make_future_dataframe(periods=len(y_test), freq='B')
    forecast = model_prophet.predict(future)
    y_pred_prophet = forecast['yhat'][-len(y_test):].values
    
    mse_prophet, rmse_prophet, mae_prophet = calculate_errors(y_test, y_pred_prophet)
    new_result = pd.DataFrame({"Ticker": [ticker], "Algorithm": ["Prophet"], "MSE": [mse_prophet], "RMSE": [rmse_prophet], "MAE": [mae_prophet]})
    if not new_result.empty:
        results = pd.concat([results, new_result.dropna(how='all')], ignore_index=True)
    
    # LSTM
    # Veriyi LSTM modeline uygun hale getirme
    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
    
    X_train = []
    y_train_LSTM = []
    for i in range(60, len(y_train_scaled)):
        X_train.append(y_train_scaled[i-60:i, 0])
        y_train_LSTM.append(y_train_scaled[i, 0])
    
    X_train, y_train_LSTM = np.array(X_train), np.array(y_train_LSTM)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model_lstm = Sequential()
    model_lstm.add(tf.keras.Input(shape=(X_train.shape[1], 1)))  # Doğru input shape
    model_lstm.add(LSTM(units=50, return_sequences=True))
    model_lstm.add(LSTM(units=50))
    model_lstm.add(Dense(1))
    
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(X_train, y_train_LSTM, epochs=100, batch_size=32)
    
    # Tahmin yapma
    inputs = y_train[len(y_train) - len(y_test) - 60:].values.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    
    X_test = []
    for i in range(60, 60 + len(y_test)):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    y_pred_lstm = model_lstm.predict(X_test)
    y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
    
    mse_lstm, rmse_lstm, mae_lstm = calculate_errors(y_test, y_pred_lstm)
    new_result = pd.DataFrame({"Ticker": [ticker], "Algorithm": ["LSTM"], "MSE": [mse_lstm], "RMSE": [rmse_lstm], "MAE": [mae_lstm]})
    if not new_result.empty:
        results = pd.concat([results, new_result.dropna(how='all')], ignore_index=True)
        
    # Grafik oluşturma
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test, label="Actual")
    plt.plot(y_test.index, y_pred_arima, label="ARIMA")
    plt.plot(y_test.index, y_pred_xgb, label="XGBoost")
    plt.plot(y_test.index, y_pred_prophet, label="Prophet")
    plt.plot(y_test.index, y_pred_lstm, label="LSTM")
    plt.title(f'{ticker} - Model Predictions')
    plt.xlabel('Date')
    plt.ylabel('Normalized Adjusted Close Price')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{ticker}_predictions.png'))
    plt.close()
    

# Sonuçları kaydetme
results.to_csv(os.path.join(output_dir, "results.csv"), index=False)

# Hata metriklerini Excel dosyasına yazma
with pd.ExcelWriter(os.path.join(output_dir, "results.xlsx")) as writer:
    results.to_excel(writer, index=False)

# Kullanılan algoritmaları kaydetme
algorithms = [
    "ARIMA",
    "XGBoost",
    "Prophet",
    "LSTM"
]

with open(os.path.join(output_dir, "algorithms.txt"), 'w') as f:
    for algo in algorithms:
        f.write(f"{algo}\n")

# Genel hata metriklerinin grafiği (tüm metrikler bir arada)
metrics = ["MSE", "RMSE", "MAE"]
mean_results = results.groupby(["Algorithm", "Ticker"])[metrics].mean().reset_index()
mean_results_melted = mean_results.melt(id_vars=["Algorithm", "Ticker"], value_vars=metrics, var_name="Metric", value_name="Value")

plt.figure(figsize=(14, 10))
sns.barplot(x="Algorithm", y="Value", hue="Ticker", data=mean_results_melted)
plt.title('Mean Error Metrics by Algorithm and Ticker')
plt.xlabel('Algorithm')
plt.ylabel('Error Metric Value')
plt.xticks(rotation=45)
plt.legend(title="Ticker", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mean_error_metrics_by_ticker.png'))
plt.close()













