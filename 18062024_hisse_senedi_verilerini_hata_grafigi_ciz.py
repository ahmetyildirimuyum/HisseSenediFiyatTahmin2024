import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sonuç dosyasını yükleme
file_path = 'D:\\UyumProjects\\UyumERP\\trunk\\duzelt\\Ahmet_YILDIRIM\\_YUKSEK_LISANS_TEZ\\_Notlar\\2024-06-18_11-46\\results.csv'
results = pd.read_csv(file_path)

# Sadece sayısal sütunları seçme
numeric_columns = results.select_dtypes(include=[np.number]).columns
mean_results = results.groupby(["Algorithm", "Ticker"])[numeric_columns].mean().reset_index()

# Genel hata metriklerinin grafiği (tüm metrikler bir arada)
metrics = ["MSE", "RMSE", "MAE"]
mean_results_melted = mean_results.melt(id_vars=["Algorithm", "Ticker"], value_vars=metrics, var_name="Metric", value_name="Value")

plt.figure(figsize=(14, 10))
sns.barplot(x="Algorithm", y="Value", hue="Ticker", data=mean_results_melted)
plt.title('Mean Error Metrics by Algorithm and Ticker')
plt.xlabel('Algorithm')
plt.ylabel('Error Metric Value')
plt.xticks(rotation=45)
plt.legend(title="Ticker", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('D:\\UyumProjects\\UyumERP\\trunk\\duzelt\\Ahmet_YILDIRIM\\_YUKSEK_LISANS_TEZ\\_Notlar\\2024-06-18_11-46\\mean_error_metrics_by_ticker.png')
plt.show()
