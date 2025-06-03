import warnings
warnings.filterwarnings("ignore")

# Import required libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import os

# Load the dataset
df = pd.read_csv("/content/pageviews-20150701-20250601.csv")
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values('ds')

# Keep only past data (avoid future leakage)
today = pd.Timestamp.now().normalize()
df = df[df['ds'] <= today]

# Create lag features
df['lag1'] = df['y'].shift(1)
df['lag7'] = df['y'].shift(7)
df['lag30'] = df['y'].shift(30)
df.dropna(inplace=True)

# Train LightGBM model
X_train = df[['lag1', 'lag7', 'lag30']]
y_train = df['y']
model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# Forecast for the next 365 days
forecast_days = 365
forecast_results = []
last_values = df['y'].tail(30).values

for i in range(forecast_days):
    lag1 = forecast_results[i-1]['y'] if i >= 1 else last_values[-1]
    lag7 = forecast_results[i-7]['y'] if i >= 7 else last_values[-7 + i] if len(last_values) >= 7 else last_values[-1]
    lag30 = forecast_results[i-30]['y'] if i >= 30 else last_values[-30 + i] if len(last_values) >= 30 else last_values[-1]

    features = np.array([[lag1, lag7, lag30]])
    y_pred = model.predict(features)[0]

    forecast_date = df['ds'].iloc[-1] + pd.Timedelta(days=i+1)
    forecast_results.append({'ds': forecast_date, 'y': max(0, y_pred)})

# Create forecast DataFrame
forecast_df = pd.DataFrame(forecast_results)
forecast_df = forecast_df.rename(columns={'y': 'LightGBM_Forecast'})

# Plot the forecast
plt.figure(figsize=(14, 6))
plt.plot(df['ds'].tail(90), df['y'].tail(90), label='Actual Data', linewidth=2)
plt.plot(forecast_df['ds'], forecast_df['LightGBM_Forecast'], label='LightGBM Forecast', linewidth=2)
plt.title("Daily Pageviews Forecast using LightGBM")
plt.xlabel("Date")
plt.ylabel("Pageviews")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
os.makedirs("charts", exist_ok=True)
plt.savefig("charts/lightgbm_forecast_extended.png")
plt.show()

# Save the forecast results to a CSV file
forecast_df.to_csv("lightgbm_forecast_2025_2026.csv", index=False)
