"""
app.py — Healthcare No-Show Prediction Dashboard
=================================================
Streamlit app that:
1. Loads the pre-trained XGBoost model from model_artifacts/
2. Accepts CSV upload of new appointments
3. Predicts risk score per patient
4. Shows interactive dashboard with KPIs, charts, risk table
5. Allows CSV download of predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="No-Show Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 28px; font-weight: 700;
        color: #1a1a2e; margin-bottom: 4px;
    }
    .sub-title {
        font-size: 14px; color: #666;
        margin-bottom: 24px;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 16px; text-align: center;
        border: 1px solid #e0e0e0;
    }
    .high-risk  { background: #FCEBEB; border-left: 4px solid #D85A30; }
    .med-risk   { background: #FAEEDA; border-left: 4px solid #EF9F27; }
    .low-risk   { background: #EAF3DE; border-left: 4px solid #1D9E75; }
    .section-header {
        font-size: 16px; font-weight: 600;
        color: #1a1a2e; margin: 20px 0 10px;
        border-bottom: 2px solid #534AB7;
        padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model artifacts ──────────────────────────────────────
@st.cache_resource
def load_model():
    base = 'model_artifacts'
    model       = joblib.load(f'{base}/noshow_model.pkl')
    feat_cols   = joblib.load(f'{base}/feature_columns.pkl')
    neigh_map   = joblib.load(f'{base}/neighbourhood_risk_map.pkl')
    with open(f'{base}/model_metadata.json') as f:
        metadata = json.load(f)
    return model, feat_cols, neigh_map, metadata

try:
    model, FEATURE_COLS, neigh_map, metadata = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model not found: {e}. Upload model_artifacts/ to your repo.")

# ── Feature engineering function ─────────────────────────────
def engineer_features(df, neigh_map):
    df = df.copy()
    df['ScheduledDay']   = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

    # Rename typos if present
    df.rename(columns={
        'Hipertension': 'Hypertension',
        'Handcap':      'Handicap',
    }, inplace=True)

    # lead_time
    df['lead_time'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    df['lead_time'] = df['lead_time'].clip(lower=0)

    # prior_noshows — default 0 for new data (no history)
    if 'prior_noshows' not in df.columns:
        df['prior_noshows'] = 0

    # Temporal
    df['day_of_week']       = df['AppointmentDay'].dt.dayofweek
    df['appointment_month'] = df['AppointmentDay'].dt.month

    # Chronic flag
    h = df.get('Hypertension', pd.Series(0, index=df.index))
    d = df.get('Diabetes',     pd.Series(0, index=df.index))
    a = df.get('Alcoholism',   pd.Series(0, index=df.index))
    df['has_chronic'] = ((h + d + a) > 0).astype(int)

    # Gender encode
    df['Gender'] = (df['Gender'] == 'M').astype(int)

    # Neighbourhood risk
    df['neighbourhood_risk'] = (
        df['Neighbourhood'].map(neigh_map).fillna(0.202)
        if 'Neighbourhood' in df.columns
        else 0.202
    )

    return df

def assign_risk_tier(score):
    if score >= 0.6:   return 'High'
    elif score >= 0.3: return 'Medium'
    else:              return 'Low'

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=60)
    st.markdown("## No-Show Predictor")
    st.markdown("---")

    st.markdown("### Upload Appointments")
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV with appointment data. Must include ScheduledDay, AppointmentDay, Age, Gender columns."
    )

    st.markdown("---")
    st.markdown("### Risk Thresholds")
    high_thresh = st.slider("High risk threshold",   0.4, 0.9, 0.6, 0.05)
    med_thresh  = st.slider("Medium risk threshold", 0.1, 0.5, 0.3, 0.05)

    st.markdown("---")
    if model_loaded:
        st.markdown("### Model Info")
        st.markdown(f"**AUC-ROC:** `{metadata['auc_roc']}`")
        st.markdown(f"**F1-Score:** `{metadata['f1_score']}`")
        st.markdown(f"**Recall:**   `{metadata['recall']}`")
        st.markdown(f"**Train size:** `{metadata['train_size']:,}`")
        st.markdown(f"**Algorithm:** XGBoost (tuned)")

    st.markdown("---")
    st.caption("Healthcare No-Show Prediction · Phase 4")

# ══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🏥 Healthcare Appointment No-Show Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload appointment data to get instant no-show risk predictions powered by XGBoost</div>', unsafe_allow_html=True)

# ── No file uploaded — show demo ──────────────────────────────
if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to start predicting. Use your MedicalAppointment.csv file.")

    # Show expected columns
    st.markdown('<div class="section-header">Expected CSV columns</div>', unsafe_allow_html=True)
    expected = pd.DataFrame({
        'Column': ['PatientId','AppointmentID','Gender','ScheduledDay',
                   'AppointmentDay','Age','Neighbourhood','Scholarship',
                   'Hypertension','Diabetes','Alcoholism','Handicap','SMS_received'],
        'Type':   ['String','String','M/F','DateTime',
                   'DateTime','Integer','String','0/1',
                   '0/1','0/1','0/1','0/1','0/1'],
        'Example':['5.50E+14','5642903','F','2016-04-29 08:07:00',
                   '2016-04-29','62','JARDIM DA PENHA','0',
                   '1','0','0','0','0']
    })
    st.dataframe(expected, use_container_width=True)

    # Show model performance
    if model_loaded:
        st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("AUC-ROC",   metadata['auc_roc'])
        c2.metric("F1-Score",  metadata['f1_score'])
        c3.metric("Precision", metadata['precision'])
        c4.metric("Recall",    metadata['recall'])
        c5.metric("Accuracy",  metadata['accuracy'])

# ── File uploaded — run predictions ───────────────────────────
else:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.success(f"✓ Loaded {len(raw_df):,} appointments")

        # Engineer features
        df_feat = engineer_features(raw_df, neigh_map)

        # Align to training feature columns
        for col in FEATURE_COLS:
            if col not in df_feat.columns:
                df_feat[col] = 0
        X = df_feat[FEATURE_COLS]

        # Predict
        risk_scores             = model.predict_proba(X)[:, 1]
        raw_df['risk_score']    = risk_scores
        raw_df['risk_tier']     = raw_df['risk_score'].apply(
            lambda s: 'High' if s >= high_thresh
                      else ('Medium' if s >= med_thresh else 'Low')
        )
        raw_df['lead_time']     = df_feat['lead_time'].values
        raw_df['risk_score_pct']= (raw_df['risk_score'] * 100).round(1)

        # ── KPI Row ───────────────────────────────────────────
        st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)
        total     = len(raw_df)
        high_n    = (raw_df['risk_tier'] == 'High').sum()
        med_n     = (raw_df['risk_tier'] == 'Medium').sum()
        low_n     = (raw_df['risk_tier'] == 'Low').sum()
        avg_risk  = raw_df['risk_score'].mean()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total appointments", f"{total:,}")
        k2.metric("High risk",   f"{high_n:,}",  delta=f"{high_n/total:.1%}", delta_color="inverse")
        k3.metric("Medium risk", f"{med_n:,}")
        k4.metric("Low risk",    f"{low_n:,}",   delta=f"{low_n/total:.1%}", delta_color="normal")
        k5.metric("Avg risk score", f"{avg_risk:.1%}")

        # ── Charts row ────────────────────────────────────────
        st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)

        # Pie chart — risk tier distribution
        with ch1:
            fig_pie = px.pie(
                raw_df, names='risk_tier',
                color='risk_tier',
                color_discrete_map={
                    'High':   '#D85A30',
                    'Medium': '#EF9F27',
                    'Low':    '#1D9E75'
                },
                title='Patient risk tier breakdown',
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        # Histogram — risk score distribution
        with ch2:
            fig_hist = px.histogram(
                raw_df, x='risk_score', nbins=30,
                title='Risk score distribution',
                color_discrete_sequence=['#534AB7'],
                labels={'risk_score': 'Risk score'}
            )
            fig_hist.add_vline(x=high_thresh, line_dash='dash',
                               line_color='#D85A30',
                               annotation_text='High threshold')
            fig_hist.add_vline(x=med_thresh,  line_dash='dash',
                               line_color='#EF9F27',
                               annotation_text='Medium threshold')
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Lead time chart ───────────────────────────────────
        st.markdown('<div class="section-header">Lead Time Analysis</div>', unsafe_allow_html=True)
        raw_df['lt_bin'] = pd.cut(
            raw_df['lead_time'],
            bins=[-1, 0, 7, 14, 30, 60, 500],
            labels=['Same-day','1–7d','8–14d','15–30d','31–60d','60d+']
        )
        lt_grp = raw_df.groupby('lt_bin', observed=True).agg(
            count=('risk_score','count'),
            avg_risk=('risk_score','mean')
        ).reset_index()

        fig_lt = px.bar(
            lt_grp, x='lt_bin', y='avg_risk',
            title='Average risk score by lead time',
            color='avg_risk',
            color_continuous_scale=['#1D9E75','#EF9F27','#D85A30'],
            labels={'lt_bin':'Lead time','avg_risk':'Avg risk score'},
            text=lt_grp['avg_risk'].map('{:.1%}'.format)
        )
        fig_lt.update_traces(textposition='outside')
        fig_lt.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_lt, use_container_width=True)

        # ── Neighbourhood risk ────────────────────────────────
        if 'Neighbourhood' in raw_df.columns:
            st.markdown('<div class="section-header">Neighbourhood Risk</div>', unsafe_allow_html=True)
            neigh_grp = (
                raw_df.groupby('Neighbourhood')['risk_score']
                .agg(['mean','count'])
                .query('count >= 5')
                .sort_values('mean', ascending=False)
                .head(15)
                .reset_index()
            )
            fig_neigh = px.bar(
                neigh_grp, x='mean', y='Neighbourhood',
                orientation='h',
                title='Top 15 neighbourhoods by avg risk score',
                color='mean',
                color_continuous_scale=['#1D9E75','#EF9F27','#D85A30'],
                labels={'mean':'Avg risk score','Neighbourhood':''},
                text=neigh_grp['mean'].map('{:.1%}'.format)
            )
            fig_neigh.update_layout(coloraxis_showscale=False, height=450)
            fig_neigh.update_traces(textposition='outside')
            st.plotly_chart(fig_neigh, use_container_width=True)

        # ── High risk patient table ───────────────────────────
        st.markdown('<div class="section-header">High Risk Patients — Action Required</div>', unsafe_allow_html=True)

        display_cols = ['risk_score_pct', 'risk_tier', 'lead_time']
        if 'PatientId'      in raw_df.columns: display_cols = ['PatientId']      + display_cols
        if 'AppointmentDay' in raw_df.columns: display_cols = display_cols + ['AppointmentDay']
        if 'Age'            in raw_df.columns: display_cols = display_cols + ['Age']
        if 'Neighbourhood'  in raw_df.columns: display_cols = display_cols + ['Neighbourhood']

        high_risk_df = (
            raw_df[raw_df['risk_tier'] == 'High']
            [display_cols]
            .sort_values('risk_score_pct', ascending=False)
            .reset_index(drop=True)
        )
        high_risk_df.columns = [c.replace('_',' ').title() for c in high_risk_df.columns]

        st.dataframe(
            high_risk_df,
            use_container_width=True,
            height=350
        )
        st.caption(f"Showing {len(high_risk_df)} high-risk patients (risk score ≥ {high_thresh:.0%})")

        # ── Full predictions download ─────────────────────────
        st.markdown('<div class="section-header">Download Predictions</div>', unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)

        with dl1:
            csv_all = raw_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇ Download all predictions CSV",
                data=csv_all,
                file_name=f"noshow_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
        with dl2:
            csv_high = high_risk_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇ Download high-risk patients only",
                data=csv_high,
                file_name=f"high_risk_patients_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )

        # ── Model performance ─────────────────────────────────
        if model_loaded:
            st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("AUC-ROC",   metadata['auc_roc'],   help="Area under ROC curve")
            m2.metric("F1-Score",  metadata['f1_score'],  help="Harmonic mean of precision and recall")
            m3.metric("Precision", metadata['precision'], help="Of predicted no-shows, how many were correct")
            m4.metric("Recall",    metadata['recall'],    help="Of actual no-shows, how many did we catch")
            m5.metric("Accuracy",  metadata['accuracy'],  help="Overall correct predictions")

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Make sure your CSV has the required columns: ScheduledDay, AppointmentDay, Age, Gender, Neighbourhood")
