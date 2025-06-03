from sklearn.neighbors import KNeighborsRegressor

# Create lag features for k-NN model
df_knn = df.copy()
df_knn['lag1'] = df_knn['y'].shift(1)
df_knn['lag7'] = df_knn['y'].shift(7)
df_knn.dropna(inplace=True)

# Train-test split for k-NN input
train_knn = df_knn.iloc[:split_index-7]
test_knn = df_knn.iloc[split_index-7:]

# Prepare input features
X_train, y_train = train_knn[['lag1', 'lag7']], train_knn['y']
X_test, y_test = test_knn[['lag1', 'lag7']], test_knn['y']

# Fit k-NN regressor
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)
forecast = model.predict(X_test)

# Store forecast results in test set
test_df.loc[:, 'k-NN'] = forecast

# Evaluate model performance
metrics = evaluate_forecast(y_test, forecast)
results['k-NN'] = metrics

# Print evaluation metrics
print("\nkNN Metrics")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(12, 5))
plt.plot(test_df['ds'], test_df['y'], label='Actual')
plt.plot(test_df['ds'], test_df['k-NN'], label='k-NN')
plt.title("kNN Forecast")
plt.legend()
plt.grid()
plt.tight_layout()

# Save forecast plot
plt.savefig("knn_forecast.png")
plt.show()
