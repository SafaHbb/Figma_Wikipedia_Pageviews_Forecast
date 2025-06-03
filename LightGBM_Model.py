from lightgbm import LGBMRegressor

# Train the LightGBM model
model = LGBMRegressor()
model.fit(X_train, y_train)

# Make predictions
forecast = model.predict(X_test)

# Save predictions to test set
test_df.loc[:, 'LightGBM'] = forecast

# Evaluate the model
metrics = evaluate_forecast(y_test, forecast)
results['LightGBM'] = metrics

# Print the metrics
print("\nLightGBM Metrics")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")

# Plot the results
plt.figure(figsize=(12, 5))
plt.plot(test_df['ds'], test_df['y'], label='Actual')
plt.plot(test_df['ds'], test_df['LightGBM'], label='LightGBM')
plt.title("LightGBM Forecast")
plt.legend()
plt.grid()
plt.tight_layout()

# Save the plot
plt.savefig("lightgbm_forecast.png")
plt.show()
