import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'PM2.5': np.random.uniform(10, 200, n_samples),
        'PM10': np.random.uniform(20, 300, n_samples),
        'NO2': np.random.uniform(5, 100, n_samples),
        'SO2': np.random.uniform(2, 80, n_samples),
        'CO': np.random.uniform(0.1, 2.0, n_samples),
        'O3': np.random.uniform(10, 120, n_samples),
    }
    df = pd.DataFrame(data)
    df['AQI'] = (
        0.5 * df['PM2.5'] + 0.3 * df['PM10'] + 0.1 * df['NO2'] +
        0.05 * df['SO2'] + 0.03 * df['CO'] * 100 + 0.02 * df['O3'] +
        np.random.normal(0, 10, n_samples)
    )
    return df

def generate_html_report(mse, r2):
    html = f"""
    <html><head><title>Air Quality Prediction Report</title></head><body>
    <h1>Air Quality Prediction Report</h1>
    <h2>Model Performance</h2>
    <table border='1'><tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Mean Squared Error</td><td>{mse:.2f}</td></tr>
    <tr><td>R2 Score</td><td>{r2:.2f}</td></tr>
    </table>
    <h2>Visualizations</h2>
    <b>Actual vs Predicted AQI:</b><br>
    <img src='aqi_prediction_results.png' width='500'><br>
    <b>Residuals Plot:</b><br>
    <img src='aqi_residuals_plot.png' width='500'><br>
    </body></html>
    """
    with open('air_quality_report.html', 'w') as f:
        f.write(html)
    print('HTML report generated: air_quality_report.html')

def generate_pdf_report(mse, r2):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Air Quality Prediction Report', ln=True, align='C')
    pdf.set_font('Arial', '', 12)
    pdf.ln(10)
    pdf.cell(0, 10, 'Model Performance:', ln=True)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(60, 10, 'Metric', 1)
    pdf.cell(60, 10, 'Value', 1)
    pdf.ln()
    pdf.set_font('Arial', '', 12)
    pdf.cell(60, 10, 'Mean Squared Error', 1)
    pdf.cell(60, 10, f'{mse:.2f}', 1)
    pdf.ln()
    pdf.cell(60, 10, 'R2 Score', 1)
    pdf.cell(60, 10, f'{r2:.2f}', 1)
    pdf.ln(15)
    pdf.cell(0, 10, 'Visualizations:', ln=True)
    if os.path.exists('aqi_prediction_results.png'):
        pdf.cell(0, 10, 'Actual vs Predicted AQI:', ln=True)
        pdf.image('aqi_prediction_results.png', w=170)
    if os.path.exists('aqi_residuals_plot.png'):
        pdf.cell(0, 10, 'Residuals Plot:', ln=True)
        pdf.image('aqi_residuals_plot.png', w=170)
    pdf.output('air_quality_report.pdf')
    print('PDF report generated: air_quality_report.pdf')

def main():
    # 1. Load data
    df = generate_synthetic_data()
    print('Sample data:')
    print(df.head())

    # 2. Preprocess data
    X = df.drop('AQI', axis=1)
    y = df['AQI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'\nModel Performance:')
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R2 Score: {r2:.2f}')

    # 5. Plot results
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title('Actual vs Predicted AQI')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout()
    plt.savefig('aqi_prediction_results.png')
    plt.show()

    # Residuals plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted AQI')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residuals Plot')
    plt.tight_layout()
    plt.savefig('aqi_residuals_plot.png')
    plt.show()

    # 6. Generate reports
    generate_html_report(mse, r2)
    generate_pdf_report(mse, r2)

    # 7. Predict new sample
    sample = pd.DataFrame({
        'PM2.5': [80], 'PM10': [120], 'NO2': [40], 'SO2': [20], 'CO': [0.8], 'O3': [60]
    })
    predicted_aqi = model.predict(sample)[0]
    print(f'\nPredicted AQI for sample input: {predicted_aqi:.2f}')

if __name__ == '__main__':
    main() 