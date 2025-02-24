import numpy as np
import pandas as pd
from prophet import Prophet

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
