from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# ARIMA model
arima_model = ARIMA(train_df['y'], order=(5, 1, 0))
arima_result = arima_model.fit()

# Forecast
arima_forecast = arima_result.forecast(steps=len(test_df))

# Evaluate
arima_metrics = evaluate_forecast(test_df['y'].values, arima_forecast)
results['ARIMA'] = arima_metrics

test_df = test_df.copy()
test_df.loc[:, 'ARIMA'] = arima_forecast.values

# Print metrics
print("\nARIMA Forecast Metrics:")
for key, val in arima_metrics.items():
    print(f"{key}: {val:.2f}")

# Plot
plt.figure(figsize=(12, 5))
plt.plot(test_df['ds'], test_df['y'], label='Actual', color='black')
plt.plot(test_df['ds'], test_df['ARIMA'], label='ARIMA Forecast', color='blue')
plt.title('ARIMA Forecast vs Actual')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("arima_forecast.png")
plt.show()
