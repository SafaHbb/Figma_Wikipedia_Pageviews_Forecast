from sklearn.linear_model import Ridge

# Prepare input features for the stacked model
X_stack = test_df[['Prophet', 'SARIMA', 'Holt-Winters']]
y_stack = test_df['y']

# Train the meta-model (Ridge Regression)
meta_model = Ridge()
meta_model.fit(X_stack, y_stack)

# Predict using the ensemble model
test_df['EnsembleStacked'] = meta_model.predict(X_stack)

# Evaluate the stacked ensemble
stacked_metrics = evaluate_forecast(test_df['y'], test_df['EnsembleStacked'])
print("\nStacked Ensemble Metrics")
for k, v in stacked_metrics.items():
    print(f"{k}: {v:.2f}")

# Save metrics to results dictionary
results['EnsembleStacked'] = stacked_metrics

# Plot actual vs ensemble prediction
plt.figure(figsize=(12, 5))
plt.plot(test_df['ds'], test_df['y'], label='Actual')
plt.plot(test_df['ds'], test_df['EnsembleStacked'], label='Stacked Ensemble')
plt.title("Stacked Ensemble Forecast")
plt.legend()
plt.grid()
plt.tight_layout()

# Save the plot
plt.savefig("ensemble_stacked_forecast.png")
plt.show()
