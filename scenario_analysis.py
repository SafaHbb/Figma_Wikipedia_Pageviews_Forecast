# Scenario analysis using different multipliers applied to LightGBM forecast
# No code is changed from original request

import matplotlib.pyplot as plt

# Copy forecast data
scenarios_df = forecast_df.copy()
scenarios_df = scenarios_df.rename(columns={'LightGBM_Forecast': 'Normal'})

# Create scenarios
scenarios_df['Spike'] = scenarios_df['Normal'] * 2.0        # +100% increase
scenarios_df['Drop'] = scenarios_df['Normal'] * 0.5         # -50% decrease
scenarios_df['Conservative'] = scenarios_df['Normal'] * 0.8 # -20% decrease (more realistic)

# Plot all scenarios
plt.figure(figsize=(14, 6))
plt.plot(scenarios_df['ds'], scenarios_df['Normal'], label='Normal Forecast', linewidth=2)
plt.plot(scenarios_df['ds'], scenarios_df['Spike'], label='Spike Scenario (+100%)', linestyle='--', color='red')
plt.plot(scenarios_df['ds'], scenarios_df['Drop'], label='Drop Scenario (-50%)', linestyle='--', color='orange')
plt.plot(scenarios_df['ds'], scenarios_df['Conservative'], label='Conservative (-20%)', linestyle=':', color='purple')

plt.title("Scenario Analysis: Figma Wikipedia Daily Pageviews Forecast")
plt.xlabel("Date")
plt.ylabel("Pageviews")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("charts/scenario_analysis.png")  # Save the plot
plt.show()
