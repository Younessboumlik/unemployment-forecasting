# Unemployment Rate Forecasting in the United States

![GitHub last commit](https://img.shields.io/github/last-commit/Younessboumlik/unemployment-forecasting)
![GitHub repo size](https://img.shields.io/github/repo-size/Younessboumlik/unemployment-forecasting)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

This project explores time series forecasting techniques to predict the US unemployment rate using historical data from January 1948 to January 2025.

## 📌 Project Overview

This study implements and compares three different modeling approaches:
1. **Polynomial Regression** (baseline model)
2. **Facebook's Prophet** (for time series with multiple seasonality)
3. **ARIMA/SARIMA** (classical statistical time series approach)

The project includes:
- Data preprocessing and exploratory analysis
- Model implementation and evaluation
- Performance comparison
- Streamlit web application for interactive visualization

## 📊 Key Findings

- ARIMA(1,1,1) showed the best performance with:
  - MAE: 0.167
  - RMSE: 0.415
  - MAPE: 2.89%
  - R²: 0.941
- Prophet model also performed well with better interpretability
- Polynomial regression proved inadequate for this complex time series





## 🚀 Usage

### Running the Streamlit App
```bash
streamlit run app.py
```

### Running Jupyter Notebooks
```bash
jupyter notebook
```

## 📈 Results Visualization

### Model Performance Comparison
| Metric | Polynomial Regression | Prophet | ARIMA |
|--------|----------------------|---------|-------|
| MAE | 0.45 | 0.323 | **0.167** |
| RMSE | 0.68 | 0.581 | **0.415** |
| MAPE | 8.12% | 6.28% | **2.89%** |
| R² | 0.218 | 0.884 | **0.941** |

### Forecast Visualizations
<div align="center">
  
**Polynomial Regression Fit**  
![Polynomial Regression](https://github.com/Younessboumlik/unemployment-forecasting/blob/main/img/regression_polynomiale.png)  
*Figure: Polynomial regression fit of degree 6 (R²=0.218)*

**Prophet Forecast Components**  
![Prophet Components](https://github.com/Younessboumlik/unemployment-forecasting/blob/main/img/prophet_components.png)  
*Figure: Prophet's decomposition of trend, seasonality and residuals*

**ARIMA(1,1,1) Forecast**  
![ARIMA Forecast](https://github.com/Younessboumlik/unemployment-forecasting/blob/main/img/arima_forecast.png)  
*Figure: ARIMA forecast with 95% confidence interval*

**ARIMA vs Prophet Forecast Comparison**  
![ARIMA vs Prophet Forecast](https://github.com/Younessboumlik/unemployment-forecasting/blob/main/img/unemployment_forecast_comparison.png)  
*Figure: Comparison of 24-month forecasts from ARIMA(1,1,1) and Prophet models*

</div>

## 📝 Report

The complete technical report (in French) is available in the repository as `ML_rapport.pdf`, containing:
- Detailed methodology
- Mathematical formulations
- Implementation details
- Performance metrics
- Discussion of results

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.


## 🙏 Acknowledgments

- **Supervisor**: Mme. Sara Baghdadi
- **Institution**: École Nationale des Sciences Appliquées, Université Sultan Moulay Slimane
- **Data Source**: [FRED Economic Data](https://fred.stlouisfed.org/series/UNRATE)
