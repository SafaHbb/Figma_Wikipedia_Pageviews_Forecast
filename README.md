# Figma Wikipedia Pageviews Forecast

This project predicts the **daily pageviews** for the Figma Wikipedia page using several time series forecasting models.

We used real data and compared classical statistical methods with machine learning models. The results include forecast charts, performance metrics, and future traffic simulations.

This project is an example of time series forecasting using data based on user behaviour.
It focuses on predicting how many people visit the Wikipedia page for Figma (software) each day and the data shows long-term trends and possible effects from real events  like product updates or news.

---

## Project Overview

### Goal:
- Forecast daily visits to the Wikipedia page of "Figma" from today untill the middle of 2026.
- Compare performance of statistical vs ML models
- Visualize forecast results
- Simulate future scenarios (e.g., spikes or drops in traffic)

---

## How to Run This Project

1. **Download the dataset**  
   The pageview data was obtained using the [Wikipedia Pageviews Analysis tool](https://pageviews.wmcloud.org/?project=en.wikipedia.org&platform=all-access&agent=user&redirects=0&range=all-time&pages=Figma_(software)).  
   - Set the **date range** to **"all-time"**  
   - Export the data as **CSV**

2. **Upload the dataset to Colab**  
   Upload the CSV file to your Colab session (typically under `/content/`).

After setup, you can run all models, evaluate results, and explore scenario forecasts.

---

## Dataset

- Source: Wikipedia pageviews from July 2015 to June 2025
- Structure: 2 columns — "date" and "pageviews"

---

## Models Used

We tested both statistical and machine learning models to predict future pageviews:

| Model Type       | Library           |
|------------------|-------------------|
| ARIMA            | statsmodels       |
| SARIMA           | statsmodels       |
| Holt-Winters     | statsmodels       |
| Prophet          | prophet           |
| k-NN             | scikit-learn      |
| LightGBM         | lightgbm          |
| Ensemble Stacked | scikit-learn      |


---

## Evaluation Metrics

Each model is evaluated on:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error

---

## Final Results

| Model            | MAE     | RMSE    | MAPE     |
|------------------|---------|---------|----------|
| LightGBM         | 146.67  | 455.62  | 119.38   |
| k-NN             | 147.37  | 982.02  | 80.10    |
| EnsembleStacked  | 397.76  | 507.09  | 453.91   |
| Holt-Winters     | 1208.27 | 1365.47 | 2482.06  |
| ARIMA            | 1203.22 | 1301.88 | 2494.22  |
| SARIMA           | 1693.45 | 1855.32 | 3466.61  |
| Prophet          | 2535.39 | 2694.41 | 5162.08  |

---

## Scenario Analysis

We use the final LightGBM forecast to simulate different traffic futures:

- **Normal**: The predicted trend
- **Spike**: +100% increase (e.g., viral news or product launch)
- **Drop**: -50% decrease (e.g., user interest drops)
- **Conservative**: -20% decrease (realistic slowdown)

These help explore edge cases, stress test models, and make decisions for content planning, budgeting, or server provisioning.

The chart is saved as:


---

## File Structure

Figma_Wikipedia_Pageviews_Forecast/
│
├── charts/ # All saved forecast and scenario plots (.png)
│
├── Loading_Splitting_Evaluation.py # Loads data, prepares train/test split, defines metrics
├── ARIMA_Model.py # ARIMA model forecast and chart
├── SARIMA_Model.py # SARIMA (seasonal ARIMA) forecast and chart
├── Prophet_Model.py # Prophet model forecast and chart
├── HoltWinters_Model.py # Holt-Winters smoothing forecast and chart
├── KNN_Model.py # k-NN regression model using lag features
├── LightGBM_Model.py # LightGBM forecast using lagged values
├── Ensemble_Stacked.py # Ridge regression combining multiple models
├── Final_Results_Plots.py # Comparison bar plots (MAE, RMSE, MAPE)
│
├── final_lightgbm_forecast.py # Full-year forecast with LightGBM
├── lightgbm_forecast_2025_2026.csv # Final LightGBM forecast output as CSV
├── scenario_analysis.py # Spike/drop scenario simulation and plot
└── README.md # Project overview and instructions


---

## Libraries 

- **pandas, numpy**: For time series data handling
- **matplotlib**: For plotting results
- **statsmodels**: Well-tested statistical forecasting models
- **prophet**: High-level time series model with trends and holidays
- **lightgbm**: Fast ML algorithm for structured data
- **scikit-learn**: Machine learning tools including KNN and Ridge regression

---



