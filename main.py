import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

np.random.seed(42)
num_samples = 1000
time = pd.date_range(start='2025-01-01', periods=num_samples, freq='h')
temperature = 20 + 5 * np.sin(np.linspace(0,10 * np.pi, num_samples)) + np.random.normal(0, 0.5, num_samples)

data = pd.DataFrame({'timestamp': time, 'temperature': temperature})
df_profet = data.rename(columns={'timestamp': 'ds', 'temperature': 'y'})

df_profet.to_excel("Data.xlsx", index=False)

model = Prophet()
model.fit(df_profet)

future = model.make_future_dataframe(periods=24, freq='h')
forecast = model.predict(future)

forecast.to_excel("PredictData.xlsx", index=False)

model.plot(forecast)

plt.figure(figsize=(12, 6))
plt.plot(df_profet['ds'], df_profet['y'], label="Dados Reais", color="blue")
plt.plot(forecast['ds'], forecast['yhat'], label="Previsão", color="red", linestyle="dashed")
plt.xlabel("Data")
plt.ylabel("Temperatura (°C)")
plt.title("Previsão de Temperatura com Prophet")
plt.legend()
plt.grid()

plt.show()
