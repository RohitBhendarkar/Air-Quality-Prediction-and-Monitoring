# AI-based Air Quality Prediction and Monitoring (Enhanced)

## Introduction
This enhanced project demonstrates how to use machine learning to predict the Air Quality Index (AQI) using both synthetic and real-world data, with advanced features for model comparison, feature engineering, hyperparameter tuning, visualization, and automated reporting.

## Enhancements
- **Real-World Dataset Support:** Automatically downloads and uses the UCI Air Quality dataset if available, otherwise falls back to synthetic data.
- **Feature Engineering:** Adds new features such as interaction terms and rolling averages.
- **Model Comparison:** Compares Linear Regression, Random Forest (with hyperparameter tuning), and XGBoost (if installed).
- **Hyperparameter Tuning:** Uses GridSearchCV for Random Forest and XGBoost.
- **Advanced Visualization:** Generates comparison plots, error distributions, and feature importance charts.
- **Automated HTML Reporting:** Creates an HTML report with results and visualizations.

## How It Works
1. **Data Loading:**
   - Tries to load a real-world CSV dataset. If not found, downloads it. If download fails, uses synthetic data.
2. **Feature Engineering:**
   - Adds new features based on available columns (e.g., temperature × humidity, rolling means).
3. **Model Training & Comparison:**
   - Trains and compares multiple models, using hyperparameter tuning for Random Forest and XGBoost.
4. **Visualization:**
   - Plots actual vs. predicted AQI, error distributions, and feature importances.
5. **Automated Reporting:**
   - Generates an HTML report (`air_quality_report.html`) with all results and plots.

## Usage
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the script:**
   ```bash
   python air_quality_prediction.py
   ```
3. **View the results:**
   - Open `air_quality_report.html` in your browser.
   - Plots are saved as PNG files in the workspace.

## Example Output
- **Model Performance Table** (in HTML report)
- **Actual vs. Predicted AQI Plot**
- **Error Distribution Plot**
- **Feature Importances Plot**

## Notes
- XGBoost is optional; if not installed, the script will skip it.
- For real-world applications, you can replace the dataset with your own CSV file (ensure it has similar columns).

## Conclusion
This enhanced project provides a robust, extensible template for air quality prediction and monitoring using AI, suitable for both demonstration and real-world adaptation. 