# Healthcare-No-Show-Analysis-Prediction

A Streamlit web application that predicts patient no-show risk using XGBoost.

## Live App
[hospital-noshow-dashboard.streamlit.app](https://hospital-noshow-dashboard.streamlit.app/)

## Project Structure
```
├── app.py                          # Main Streamlit dashboard
├── train_model.py                  # Model training script (run in Colab)
├── requirements.txt                # Python dependencies
├── model_artifacts/
│   ├── noshow_model.pkl            # Trained XGBoost model
│   ├── feature_columns.pkl         # Feature column order
│   ├── neighbourhood_risk_map.pkl  # Neighbourhood encoding map
│   └── model_metadata.json         # Model metrics
└── README.md
```

## How to Use
1. Open the app URL
2. Upload `MedicalAppointment.csv`
3. View risk scores, KPIs, and charts
4. Download predictions CSV

## Model
- Algorithm: XGBoost (tuned with RandomizedSearchCV)
- Target: No-show prediction (binary)
- Key features: lead_time, prior_noshows, neighbourhood_risk
- Evaluation: AUC-ROC, F1-score, Recall

## Dataset
- 110,527 medical appointments from Vitória, Brazil
- 20.2% no-show rate
- Source: Kaggle Medical Appointment No-Shows
# Healthcare Appointment No-Show Prediction Dashboard

A Streamlit web application that predicts patient no-show risk using XGBoost.

## Live App
[hospital-noshow-dashboard.streamlit.app](https://hospital-noshow-dashboard.streamlit.app/)

## Project Structure
```
├── app.py                          # Main Streamlit dashboard
├── train_model.py                  # Model training script (run in Colab)
├── requirements.txt                # Python dependencies
├── model_artifacts/
│   ├── noshow_model.pkl            # Trained XGBoost model
│   ├── feature_columns.pkl         # Feature column order
│   ├── neighbourhood_risk_map.pkl  # Neighbourhood encoding map
│   └── model_metadata.json         # Model metrics
└── README.md
```

## How to Use
1. Open the app URL
2. Upload `MedicalAppointment.csv`
3. View risk scores, KPIs, and charts
4. Download predictions CSV

## Model
- Algorithm: XGBoost (tuned with RandomizedSearchCV)
- Target: No-show prediction (binary)
- Key features: lead_time, prior_noshows, neighbourhood_risk
- Evaluation: AUC-ROC, F1-score, Recall

## Dataset
- 110,527 medical appointments from Vitória, Brazil
- 20.2% no-show rate
- Source: Kaggle Medical Appointment No-Shows
