# Labour Market Prediction

A Streamlit dashboard for Maharashtra labour-market forecasts, training-seat recommendations, and skill shortages.

## Setup

Use Python 3.10 or newer. Install the declared dependencies and start the dashboard:

```bash
python -m pip install -r requirements.txt
streamlit run frontend.py
```

The default `LMIS_SOURCE=lmis` mode reads `maharashtra_lmis_sample.csv` and runs the forecast model. This included CSV is synthetic demonstration data, not official labour-market data.

## Data sources

Set `LMIS_SOURCE` to select a source:

- `lmis` (default): the sample CSV and the model in `lmis_model.py`.
- `mock`: generated demonstration data.
- `csv`: `series.csv`, `summary.csv`, and optionally `skills.csv` in `LMIS_DATA_DIR` (default `data_real`); optional metadata can be supplied in `meta.json`.
- `api`: the `/series`, `/summary`, and optional `/skills` endpoints at `LMIS_API_URL` (default `http://localhost:8000`). Set `LMIS_API_KEY` when the API requires an `X-API-Key` header.

The CSV and API sources must provide the required columns documented in `adapters.py`. The `data_real` files and an API service are not included in this repository.

## Tests

No automated test files are currently included. Run `python -m pytest` to check for discovered tests after adding them.
## Data Visualisation
A clean graph is used to visualise the demand and supply of different trades 

## Machine Learning models 
Task                                          Algorithm                 
- Baseline demand trend                       Linear Regression
- Demand forecast, one trade                  SARIMA / Prophet
- Demand forecast, many trades together       XBoost / LightGBM on lag features
- Bloom / stable / decline label              Random Forest or XGBoost classifier
- Supply projection                           Cohort model (graduates in -  retirements/attrition) + regression on enrolment trend
- Emerging skill discovery                    K-Means / BERTopic

## 📈 AI-Powered Skill Recommendations

- Predictive Shortage Modeling: Utilizes ML models to forecast upcoming skill shortages and labor market gaps before they affect business     growth.
- Strategic Investment Recommendations: Suggests precise areas where educational institutions, companies, and policymakers should direct      training and capital investments.
