import matplotlib.pyplot as plt
import pandas as pd

# Create full metrics DataFrame
metrics_df = pd.DataFrame(results).T.sort_values('MAPE')
print("\n All Forecasting Models Summary:")
print(metrics_df.round(2))

# Plotting function
def plot_metric_bar(metric_name):
    plt.figure(figsize=(10, 5))
    metrics_df[metric_name].sort_values().plot(kind='bar', color='skyblue')
    plt.title(f"{metric_name} Comparison Across Models")
    plt.ylabel(metric_name)
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"charts/{metric_name}_comparison.png")

    plt.show()

# Plot each metric
plot_metric_bar('MAE')
plot_metric_bar('RMSE')
plot_metric_bar('MAPE')
