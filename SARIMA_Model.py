from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# SARIMA model
model = SARIMAX(train_df['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
result = model.fit(disp=False)

# Forecast future values
forecast = result.forecast(steps=len(test_df))

# Add forecast to test DataFrame
test_df.loc[:, 'SARIMA'] = forecast.values

# Evaluate forecast
metrics = evaluate_forecast(test_df['y'], test_df['SARIMA'])
results['SARIMA'] = metrics

# Print evaluation metrics
print("\nSARIMA Metrics")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")

# Plot forecast vs actual
plt.figure(figsize=(12, 5))
plt.plot(test_df['ds'], test_df['y'], label='Actual')
plt.plot(test_df['ds'], test_df['SARIMA'], label='SARIMA')
plt.title("SARIMA Forecast")
plt.legend()
plt.grid()
plt.tight_layout()

# Save plot to file
plt.savefig("sarima_forecast.png")
plt.show()
