from prophet import Prophet

# Rename columns for Prophet compatibility
prophet_train = train_df.rename(columns={'ds': 'ds', 'y': 'y'})

# Initialize and fit Prophet model
model = Prophet(daily_seasonality=True, weekly_seasonality=True)
model.fit(prophet_train)

# Create future dataframe for prediction
future = model.make_future_dataframe(periods=len(test_df))
forecast = model.predict(future)

# Extract and align forecast results
pred = forecast[['ds', 'yhat']].set_index('ds').loc[test_df['ds']]
test_df.loc[:, 'Prophet'] = pred['yhat'].values

# Evaluate forecast accuracy
metrics = evaluate_forecast(test_df['y'], test_df['Prophet'])
results['Prophet'] = metrics

# Print evaluation metrics
print("\nProphet Metrics")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")

# Plot actual vs forecast values
plt.figure(figsize=(12, 5))
plt.plot(test_df['ds'], test_df['y'], label='Actual')
plt.plot(test_df['ds'], test_df['Prophet'], label='Prophet')
plt.title("Prophet Forecast")
plt.legend()
plt.grid()
plt.tight_layout()

# Save the forecast plot
plt.savefig("prophet_forecast.png")
plt.show()
