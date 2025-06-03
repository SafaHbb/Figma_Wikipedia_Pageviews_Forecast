from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit Holt-Winters seasonal model (additive trend and seasonality)
model = ExponentialSmoothing(train_df['y'], trend='add', seasonal='add', seasonal_periods=7)
result = model.fit()

# Forecast future values
forecast = result.forecast(len(test_df))

# Add forecast to test DataFrame
test_df.loc[:, 'Holt-Winters'] = forecast.values

# Evaluate forecast accuracy
metrics = evaluate_forecast(test_df['y'], test_df['Holt-Winters'])
results['Holt-Winters'] = metrics

# Print evaluation metrics
print("\nHolt-Winters Metrics")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")

# Plot actual vs forecast values
plt.figure(figsize=(12, 5))
plt.plot(test_df['ds'], test_df['y'], label='Actual')
plt.plot(test_df['ds'], test_df['Holt-Winters'], label='Holt-Winters')
plt.title("Holt-Winters Forecast")
plt.legend()
plt.grid()
plt.tight_layout()

# Save the forecast plot
plt.savefig("holtwinters_forecast.png")
plt.show()
