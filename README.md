# Weather Trend Forecasting

This project analyzes the Kaggle Global Weather Repository dataset and forecasts future weather trends. It includes data cleaning, exploratory data analysis, forecasting models, anomaly detection, feature importance, air quality correlation, and a simple Streamlit dashboard.

## PM Accelerator Mission

PM Accelerator's mission is to provide access to education and career growth opportunities. This project includes that mission in the dashboard/report as requested by the assessment.

## Dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code

Place the CSV file here:

```text
data/raw/GlobalWeatherRepository.csv
```

If your downloaded file has a different name, either rename it to `GlobalWeatherRepository.csv` or update `DATA_PATH` in `app/streamlit_app.py`.

## Project Structure

```text
weather-trend-forecasting/
  app/
    streamlit_app.py
  data/
    raw/
    processed/
  notebooks/
    README.md
  reports/
    final_report_template.md
    figures/
  src/
    data_processing.py
    modeling.py
    visualization.py
  .gitignore
  README.md
  requirements.txt
```

## Methods

- Data cleaning and preprocessing
- Missing value handling
- Outlier detection
- Exploratory data analysis
- Temperature and precipitation visualizations
- Time series forecasting using `last_updated`
- Baseline, Random Forest, Gradient Boosting, and ensemble models
- Anomaly detection using Isolation Forest
- Feature importance analysis
- Air quality and weather correlation analysis
- Spatial/geographical analysis

## How To Run

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Recommended Workflow

1. Download the Kaggle dataset.
2. Put the CSV in `data/raw/GlobalWeatherRepository.csv`.
3. Run the Streamlit app.
4. Review the EDA charts.
5. Train and compare forecasting models.
6. Export screenshots/charts into `reports/figures/`.
7. Fill in `reports/final_report_template.md`.
8. Push the project to GitHub.

## Model Evaluation

The dashboard reports:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- R2 score

## GitHub Submission

Push this folder to GitHub:

```bash
git init
git add .
git commit -m "Add weather trend forecasting project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/weather-trend-forecasting.git
git push -u origin main
```

Make the repository public, or invite:

```text
community@pmaccelerator.io
hr@pmaccelerator.io
```

## Demo Video Talking Points

- Explain the project objective.
- Show where the dataset is loaded.
- Show the data cleaning steps.
- Show EDA visualizations.
- Show model comparison metrics.
- Show anomaly detection and feature importance.
- Show the dashboard output.
