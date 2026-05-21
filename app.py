"""
train_model.py — Train XGBoost model and save all artifacts
Run once: python train_model.py
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ── Load and clean ─────────────────────────────────────────────
df = pd.read_csv('MedicalAppointment.csv')
df = df[df['Age'] >= 0].copy()
df.rename(columns={'Hipertension':'Hypertension','Handcap':'Handicap','No-show':'NoShow'}, inplace=True)

# ── Feature Engineering ────────────────────────────────────────
df['ScheduledDay']   = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['lead_time']          = (df['AppointmentDay'] - df['ScheduledDay']).dt.days.clip(lower=0)
df['day_of_week']        = df['AppointmentDay'].dt.dayofweek
df['appointment_month']  = df['AppointmentDay'].dt.month
df['has_chronic']        = ((df['Hypertension'] + df['Diabetes'] + df['Alcoholism']) > 0).astype(int)
df['Gender']             = (df['Gender'] == 'M').astype(int)

# Neighbourhood risk map
neigh_map = df.groupby('Neighbourhood')['NoShow'].apply(
    lambda x: (x == 'Yes').mean()
).to_dict()
joblib.dump(neigh_map, 'model_artifacts/neighbourhood_risk_map.pkl')

df['neighbourhood_risk'] = df['Neighbourhood'].map(neigh_map).fillna(0.202)
df['prior_noshows']      = 0  # no history available
df['target']             = (df['NoShow'] == 'Yes').astype(int)

FEATURE_COLS = [
    'lead_time','day_of_week','appointment_month',
    'Age','Gender','SMS_received','Scholarship',
    'Hypertension','Diabetes','Alcoholism','Handicap',
    'has_chronic','neighbourhood_risk','prior_noshows'
]
joblib.dump(FEATURE_COLS, 'model_artifacts/feature_columns.pkl')

X = df[FEATURE_COLS]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Train XGBoost
model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='logloss',
    random_state=42
)
model.fit(X_res, y_res)
joblib.dump(model, 'model_artifacts/noshow_model.pkl')

# Metrics
y_prob = model.predict_proba(X_test)[:,1]
y_pred = (y_prob >= 0.5).astype(int)

metadata = {
    "auc_roc":   round(roc_auc_score(y_test, y_prob), 3),
    "f1_score":  round(f1_score(y_test, y_pred), 3),
    "precision": round(precision_score(y_test, y_pred), 3),
    "recall":    round(recall_score(y_test, y_pred), 3),
    "accuracy":  round(accuracy_score(y_test, y_pred), 3),
    "train_size": len(X_res),
    "features":  FEATURE_COLS
}
with open('model_artifacts/model_metadata.json','w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Model trained and saved!")
print(f"   AUC-ROC : {metadata['auc_roc']}")
print(f"   F1-Score: {metadata['f1_score']}")
print(f"   Recall  : {metadata['recall']}")
print(f"   Accuracy: {metadata['accuracy']}")"""
train_model.py — Train XGBoost model and save all artifacts
Run once: python train_model.py
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ── Load and clean ─────────────────────────────────────────────
df = pd.read_csv('MedicalAppointment.csv')
df = df[df['Age'] >= 0].copy()
df.rename(columns={'Hipertension':'Hypertension','Handcap':'Handicap','No-show':'NoShow'}, inplace=True)

# ── Feature Engineering ────────────────────────────────────────
df['ScheduledDay']   = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['lead_time']          = (df['AppointmentDay'] - df['ScheduledDay']).dt.days.clip(lower=0)
df['day_of_week']        = df['AppointmentDay'].dt.dayofweek
df['appointment_month']  = df['AppointmentDay'].dt.month
df['has_chronic']        = ((df['Hypertension'] + df['Diabetes'] + df['Alcoholism']) > 0).astype(int)
df['Gender']             = (df['Gender'] == 'M').astype(int)

# Neighbourhood risk map
neigh_map = df.groupby('Neighbourhood')['NoShow'].apply(
    lambda x: (x == 'Yes').mean()
).to_dict()
joblib.dump(neigh_map, 'model_artifacts/neighbourhood_risk_map.pkl')

df['neighbourhood_risk'] = df['Neighbourhood'].map(neigh_map).fillna(0.202)
df['prior_noshows']      = 0  # no history available
df['target']             = (df['NoShow'] == 'Yes').astype(int)

FEATURE_COLS = [
    'lead_time','day_of_week','appointment_month',
    'Age','Gender','SMS_received','Scholarship',
    'Hypertension','Diabetes','Alcoholism','Handicap',
    'has_chronic','neighbourhood_risk','prior_noshows'
]
joblib.dump(FEATURE_COLS, 'model_artifacts/feature_columns.pkl')

X = df[FEATURE_COLS]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Train XGBoost
model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='logloss',
    random_state=42
)
model.fit(X_res, y_res)
joblib.dump(model, 'model_artifacts/noshow_model.pkl')

# Metrics
y_prob = model.predict_proba(X_test)[:,1]
y_pred = (y_prob >= 0.5).astype(int)

metadata = {
    "auc_roc":   round(roc_auc_score(y_test, y_prob), 3),
    "f1_score":  round(f1_score(y_test, y_pred), 3),
    "precision": round(precision_score(y_test, y_pred), 3),
    "recall":    round(recall_score(y_test, y_pred), 3),
    "accuracy":  round(accuracy_score(y_test, y_pred), 3),
    "train_size": len(X_res),
    "features":  FEATURE_COLS
}
with open('model_artifacts/model_metadata.json','w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Model trained and saved!")
print(f"   AUC-ROC : {metadata['auc_roc']}")
print(f"   F1-Score: {metadata['f1_score']}")
print(f"   Recall  : {metadata['recall']}")
print(f"   Accuracy: {metadata['accuracy']}")
