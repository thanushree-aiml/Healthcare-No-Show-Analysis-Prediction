"""
app.py — Healthcare No-Show Prediction Dashboard
=================================================
Features:
  1. Loads pre-trained XGBoost model from model_artifacts/
  2. Accepts CSV upload of appointments
  3. Predicts risk score per patient
  4. Shows KPIs, charts, full prediction table, risk table
  5. Simulated SMS notification panel for high-risk patients
  6. Download buttons for all predictions and high-risk CSV
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare No-Show Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* General */
    .block-container { padding-top: 1.5rem; }

    /* Title */
    .main-title {
        font-size: 30px; font-weight: 700;
        color: #0D1B4B; margin-bottom: 2px;
    }
    .sub-title {
        font-size: 14px; color: #555;
        margin-bottom: 20px;
    }

    /* Section header */
    .section-header {
        font-size: 17px; font-weight: 700;
        color: #0D1B4B; margin: 24px 0 10px;
        padding-bottom: 6px;
        border-bottom: 3px solid #0077B6;
    }

    /* Risk badges */
    .badge-high   { background:#FCEBEB; color:#C0392B; padding:3px 10px;
                    border-radius:20px; font-weight:600; font-size:12px; }
    .badge-medium { background:#FAEEDA; color:#D68910; padding:3px 10px;
                    border-radius:20px; font-weight:600; font-size:12px; }
    .badge-low    { background:#EAF3DE; color:#1D8348; padding:3px 10px;
                    border-radius:20px; font-weight:600; font-size:12px; }

    /* SMS card */
    .sms-card {
        background: #F0FFF4;
        border: 1px solid #A9DFBF;
        border-left: 5px solid #1D9E75;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
        font-size: 13px;
    }
    .sms-card-sent {
        background: #EBF5FB;
        border: 1px solid #AED6F1;
        border-left: 5px solid #0077B6;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
        font-size: 13px;
    }

    /* Prediction table styling */
    .pred-table { font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ── Load model artifacts ──────────────────────────────────────
@st.cache_resource
def load_model():
    base = 'model_artifacts'
    model      = joblib.load(f'{base}/noshow_model.pkl')
    feat_cols  = joblib.load(f'{base}/feature_columns.pkl')
    neigh_map  = joblib.load(f'{base}/neighbourhood_risk_map.pkl')
    with open(f'{base}/model_metadata.json') as f:
        metadata = json.load(f)
    return model, feat_cols, neigh_map, metadata

try:
    model, FEATURE_COLS, neigh_map, metadata = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Model artifacts not found: {e}")
    st.stop()


# ── Feature engineering ───────────────────────────────────────
def engineer_features(df, neigh_map):
    df = df.copy()

    # Rename common typos
    df.rename(columns={
        'Hipertension': 'Hypertension',
        'Handcap':      'Handicap',
        'No-show':      'NoShow',
    }, inplace=True)

    df['ScheduledDay']   = pd.to_datetime(df['ScheduledDay'],   errors='coerce')
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], errors='coerce')

    df['lead_time']         = (df['AppointmentDay'] - df['ScheduledDay']).dt.days.clip(lower=0)
    df['day_of_week']       = df['AppointmentDay'].dt.dayofweek
    df['appointment_month'] = df['AppointmentDay'].dt.month

    h = df.get('Hypertension', pd.Series(0, index=df.index))
    d = df.get('Diabetes',     pd.Series(0, index=df.index))
    a = df.get('Alcoholism',   pd.Series(0, index=df.index))
    df['has_chronic'] = ((h + d + a) > 0).astype(int)

    df['Gender'] = (df['Gender'].astype(str).str.upper() == 'M').astype(int)

    df['neighbourhood_risk'] = (
        df['Neighbourhood'].map(neigh_map).fillna(0.202)
        if 'Neighbourhood' in df.columns else 0.202
    )
    if 'prior_noshows' not in df.columns:
        df['prior_noshows'] = 0

    return df


# ── SMS message generator ─────────────────────────────────────
def generate_sms(row):
    appt_date = str(row.get('AppointmentDay', 'your upcoming appointment'))[:10]
    patient_id = str(row.get('PatientId', 'N/A'))[:8]
    neighbourhood = row.get('Neighbourhood', 'your area')
    risk_pct = round(row['risk_score'] * 100, 1)
    return (
        f"Dear Patient (ID: ...{patient_id[-4:]}), this is a reminder for your medical "
        f"appointment on {appt_date} at {neighbourhood} clinic. "
        f"Our system has flagged your appointment as high priority. "
        f"Please confirm your attendance or reschedule at least 48 hours in advance. "
        f"Reply YES to confirm or call 1800-CLINIC to reschedule. "
        f"[Risk Score: {risk_pct}%]"
    )


# ── Waiting band helper ───────────────────────────────────────
def waiting_band(d):
    if d == 0:    return 'Same Day'
    elif d <= 7:  return '1–7 Days'
    elif d <= 14: return '8–14 Days'
    elif d <= 30: return '15–30 Days'
    elif d <= 60: return '31–60 Days'
    else:         return '60+ Days'


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 No-Show Predictor")
    st.markdown("---")

    st.markdown("### 📂 Upload Appointments")
    uploaded_file = st.file_uploader(
        "Upload CSV file", type=['csv'],
        help="Upload MedicalAppointment.csv or any appointment CSV with required columns."
    )

    st.markdown("---")
    st.markdown("### ⚙️ Risk Thresholds")
    high_thresh = st.slider("High risk (≥)", 0.4, 0.9, 0.5, 0.05,
                            help="Patients above this score are High risk")
    med_thresh  = st.slider("Medium risk (≥)", 0.1, 0.5, 0.25, 0.05,
                            help="Patients above this score are Medium risk")

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown(f"**AUC-ROC:** `{metadata['auc_roc']}`")
    st.markdown(f"**F1-Score:** `{metadata['f1_score']}`")
    st.markdown(f"**Recall:** `{metadata['recall']}`")
    st.markdown(f"**Accuracy:** `{metadata['accuracy']}`")
    st.markdown(f"**Algorithm:** XGBoost (SMOTE balanced)")
    st.markdown(f"**Train size:** `{metadata['train_size']:,}`")

    st.markdown("---")
    st.markdown("### 📱 SMS Settings")
    sms_auto_send = st.checkbox("Auto-send to all High Risk", value=False)
    sms_limit     = st.slider("Max SMS to send", 5, 50, 10,
                              help="Limit how many SMS reminders to send at once")

    st.markdown("---")
    st.caption("Healthcare No-Show Prediction · AI-Data Analytics Internship")


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🏥 Healthcare Appointment No-Show Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload appointment data → get instant XGBoost risk predictions → send SMS reminders to high-risk patients</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# NO FILE — SHOW INSTRUCTIONS
# ══════════════════════════════════════════════════════════════
if uploaded_file is None:
    st.info("👈 **Upload your MedicalAppointment.csv** from the sidebar to start predictions.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Expected CSV Columns</div>', unsafe_allow_html=True)
        expected = pd.DataFrame({
            'Column':  ['PatientId','AppointmentID','Gender','ScheduledDay',
                        'AppointmentDay','Age','Neighbourhood','Scholarship',
                        'Hypertension','Diabetes','Alcoholism','Handicap','SMS_received'],
            'Type':    ['String','String','M / F','DateTime',
                        'DateTime','Integer','String','0 / 1',
                        '0 / 1','0 / 1','0 / 1','0 / 1','0 / 1'],
        })
        st.dataframe(expected, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("AUC-ROC",   metadata['auc_roc'])
        m1.metric("F1-Score",  metadata['f1_score'])
        m1.metric("Recall",    metadata['recall'])
        m2.metric("Precision", metadata['precision'])
        m2.metric("Accuracy",  metadata['accuracy'])
        m2.metric("Train Size",f"{metadata['train_size']:,}")

    st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.success("**Step 1**\nUpload your appointment CSV using the sidebar")
    s2.info("**Step 2**\nXGBoost model predicts no-show risk score per patient")
    s3.warning("**Step 3**\nView predictions, charts, and high-risk patient table")
    s4.success("**Step 4**\nSend SMS reminders + download prediction CSV")

    st.stop()


# ══════════════════════════════════════════════════════════════
# FILE UPLOADED — RUN PIPELINE
# ══════════════════════════════════════════════════════════════
try:
    raw_df = pd.read_csv(uploaded_file)

    with st.spinner("🔄 Running predictions..."):
        time.sleep(0.5)

        # Engineer features
        df_feat = engineer_features(raw_df, neigh_map)

        # Align to training features
        for col in FEATURE_COLS:
            if col not in df_feat.columns:
                df_feat[col] = 0
        X = df_feat[FEATURE_COLS].fillna(0)

        # Predict
        risk_scores = model.predict_proba(X)[:, 1]

        # Add predictions back to original df
        result_df = raw_df.copy()
        result_df['Risk Score (%)'] = (risk_scores * 100).round(2)
        result_df['Risk Score Raw'] = risk_scores.round(4)
        result_df['Risk Tier'] = result_df['Risk Score Raw'].apply(
            lambda s: 'High' if s >= high_thresh
            else ('Medium' if s >= med_thresh else 'Low')
        )
        result_df['Lead Time (Days)'] = df_feat['lead_time'].values
        result_df['Waiting Band']     = result_df['Lead Time (Days)'].apply(waiting_band)
        result_df['Predicted No-Show'] = (risk_scores >= high_thresh).map(
            {True: 'YES — Likely No-Show', False: 'NO — Likely to Attend'}
        )
        if 'NoShow' in df_feat.columns or 'No-show' in raw_df.columns:
            actual_col = 'No-show' if 'No-show' in raw_df.columns else 'NoShow'
            result_df['Actual No-Show'] = raw_df[actual_col].values

    st.success(f"✅ Predictions complete for **{len(result_df):,}** appointments!")

    # ── KPI Row ───────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Overview KPIs</div>', unsafe_allow_html=True)

    total    = len(result_df)
    high_n   = (result_df['Risk Tier'] == 'High').sum()
    med_n    = (result_df['Risk Tier'] == 'Medium').sum()
    low_n    = (result_df['Risk Tier'] == 'Low').sum()
    avg_risk = risk_scores.mean()
    pred_noshow_pct = high_n / total * 100

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Appointments", f"{total:,}")
    k2.metric("🔴 High Risk",    f"{high_n:,}",  f"{high_n/total:.1%}", delta_color="inverse")
    k3.metric("🟡 Medium Risk",  f"{med_n:,}",   f"{med_n/total:.1%}")
    k4.metric("🟢 Low Risk",     f"{low_n:,}",   f"{low_n/total:.1%}", delta_color="normal")
    k5.metric("Avg Risk Score",  f"{avg_risk:.1%}")
    k6.metric("Predicted No-Show Rate", f"{pred_noshow_pct:.1f}%")

    # ══════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Charts",
        "📋 All Predictions",
        "🔴 High Risk Patients",
        "📱 SMS Notifications",
        "⬇️ Download"
    ])

    # ── TAB 1: CHARTS ─────────────────────────────────────────
    with tab1:

        # Row 1
        c1, c2 = st.columns(2)

        with c1:
            # Donut — risk tier split
            tier_counts = result_df['Risk Tier'].value_counts().reset_index()
            tier_counts.columns = ['Risk Tier', 'Count']
            fig_donut = px.pie(
                tier_counts, names='Risk Tier', values='Count',
                color='Risk Tier',
                color_discrete_map={'High':'#D85A30','Medium':'#EF9F27','Low':'#1D9E75'},
                title='Patient Risk Tier Breakdown',
                hole=0.45
            )
            fig_donut.update_traces(textposition='inside', textinfo='percent+label')
            fig_donut.update_layout(showlegend=True, legend=dict(orientation='h', y=-0.1))
            st.plotly_chart(fig_donut, use_container_width=True)

        with c2:
            # Histogram — risk score
            fig_hist = px.histogram(
                result_df, x='Risk Score (%)', nbins=40,
                title='Risk Score Distribution',
                color_discrete_sequence=['#0077B6'],
                labels={'Risk Score (%)': 'Risk Score (%)'}
            )
            fig_hist.add_vline(x=high_thresh*100, line_dash='dash',
                               line_color='#D85A30', annotation_text=f'High ≥{high_thresh*100:.0f}%')
            fig_hist.add_vline(x=med_thresh*100, line_dash='dash',
                               line_color='#EF9F27', annotation_text=f'Medium ≥{med_thresh*100:.0f}%')
            fig_hist.update_layout(bargap=0.05)
            st.plotly_chart(fig_hist, use_container_width=True)

        # Row 2
        c3, c4 = st.columns(2)

        with c3:
            # Lead time vs risk
            lt_grp = result_df.groupby('Waiting Band').agg(
                Count=('Risk Score Raw','count'),
                Avg_Risk=('Risk Score Raw','mean')
            ).reset_index()
            order = ['Same Day','1–7 Days','8–14 Days','15–30 Days','31–60 Days','60+ Days']
            lt_grp['Waiting Band'] = pd.Categorical(lt_grp['Waiting Band'], categories=order, ordered=True)
            lt_grp = lt_grp.sort_values('Waiting Band')
            lt_grp['Avg Risk %'] = (lt_grp['Avg_Risk'] * 100).round(1)

            fig_lt = px.bar(
                lt_grp, x='Waiting Band', y='Avg Risk %',
                title='Average Risk Score by Waiting Period',
                color='Avg Risk %',
                color_continuous_scale=['#1D9E75','#EF9F27','#D85A30'],
                text='Avg Risk %'
            )
            fig_lt.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_lt.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_lt, use_container_width=True)

        with c4:
            # Gender split
            if 'Gender' in raw_df.columns:
                gender_map = {'F':'Female','M':'Male'}
                result_df['Gender Label'] = raw_df['Gender'].map(gender_map).fillna('Unknown')
                gen_grp = result_df.groupby('Gender Label')['Risk Score Raw'].mean().reset_index()
                gen_grp.columns = ['Gender','Avg Risk Score']
                gen_grp['Avg Risk %'] = (gen_grp['Avg Risk Score']*100).round(1)

                fig_gen = px.bar(
                    gen_grp, x='Gender', y='Avg Risk %',
                    title='Average Risk Score by Gender',
                    color='Gender',
                    color_discrete_map={'Female':'#0077B6','Male':'#E05C2A'},
                    text='Avg Risk %'
                )
                fig_gen.update_traces(texttemplate='%{text}%', textposition='outside')
                fig_gen.update_layout(showlegend=False)
                st.plotly_chart(fig_gen, use_container_width=True)

        # Row 3 — Neighbourhood
        if 'Neighbourhood' in result_df.columns:
            neigh_grp = (
                result_df.groupby('Neighbourhood')
                .agg(Count=('Risk Score Raw','count'), Avg_Risk=('Risk Score Raw','mean'))
                .query('Count >= 5')
                .sort_values('Avg_Risk', ascending=False)
                .head(15)
                .reset_index()
            )
            neigh_grp['Avg Risk %'] = (neigh_grp['Avg_Risk']*100).round(1)

            fig_neigh = px.bar(
                neigh_grp, x='Avg Risk %', y='Neighbourhood',
                orientation='h',
                title='Top 15 Neighbourhoods by Average Risk Score',
                color='Avg Risk %',
                color_continuous_scale=['#1D9E75','#EF9F27','#D85A30'],
                text='Avg Risk %'
            )
            fig_neigh.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_neigh.update_layout(coloraxis_showscale=False, height=480,
                                    yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_neigh, use_container_width=True)

        # Age group chart
        if 'Age' in result_df.columns:
            def age_group(a):
                if a < 12:  return 'Child (0-11)'
                elif a < 18: return 'Teen (12-17)'
                elif a < 36: return 'Young Adult (18-35)'
                elif a < 60: return 'Adult (36-59)'
                else:        return 'Senior (60+)'

            result_df['Age Group'] = result_df['Age'].apply(age_group)
            age_grp = result_df.groupby('Age Group')['Risk Score Raw'].mean().reset_index()
            age_grp.columns = ['Age Group','Avg Risk']
            age_grp['Avg Risk %'] = (age_grp['Avg Risk']*100).round(1)
            order_age = ['Child (0-11)','Teen (12-17)','Young Adult (18-35)','Adult (36-59)','Senior (60+)']
            age_grp['Age Group'] = pd.Categorical(age_grp['Age Group'], categories=order_age, ordered=True)
            age_grp = age_grp.sort_values('Age Group')

            fig_age = px.bar(
                age_grp, x='Age Group', y='Avg Risk %',
                title='Average Risk Score by Age Group',
                color='Avg Risk %',
                color_continuous_scale=['#1D9E75','#EF9F27','#D85A30'],
                text='Avg Risk %'
            )
            fig_age.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_age.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_age, use_container_width=True)

    # ── TAB 2: ALL PREDICTIONS ────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Complete Prediction Results</div>', unsafe_allow_html=True)

        # Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            tier_filter = st.multiselect(
                "Filter by Risk Tier",
                ['High','Medium','Low'],
                default=['High','Medium','Low']
            )
        with f2:
            sort_col = st.selectbox(
                "Sort by",
                ['Risk Score (%)','Lead Time (Days)','Age'],
                index=0
            )
        with f3:
            sort_order = st.radio("Order", ['Descending','Ascending'], horizontal=True)

        # Build display columns
        show_cols = []
        for c in ['PatientId','AppointmentID','AppointmentDay','Age','Neighbourhood',
                  'Gender','SMS_received','Lead Time (Days)','Waiting Band',
                  'Risk Score (%)','Risk Tier','Predicted No-Show']:
            if c in result_df.columns:
                show_cols.append(c)
        if 'Actual No-Show' in result_df.columns:
            show_cols.append('Actual No-Show')

        filtered_df = result_df[result_df['Risk Tier'].isin(tier_filter)][show_cols]
        ascending   = sort_order == 'Ascending'
        if sort_col in filtered_df.columns:
            filtered_df = filtered_df.sort_values(sort_col, ascending=ascending)
        filtered_df = filtered_df.reset_index(drop=True)

        st.markdown(f"**Showing {len(filtered_df):,} of {total:,} records**")

        # Color code the Risk Tier column
        def color_tier(val):
            if val == 'High':   return 'background-color: #FCEBEB; color: #C0392B; font-weight:bold'
            elif val == 'Medium': return 'background-color: #FAEEDA; color: #D68910; font-weight:bold'
            else:               return 'background-color: #EAF3DE; color: #1D8348; font-weight:bold'

        def color_prediction(val):
            if 'YES' in str(val): return 'background-color: #FCEBEB; color: #C0392B; font-weight:bold'
            return 'background-color: #EAF3DE; color: #1D8348'

        styled = filtered_df.style \
            .applymap(color_tier, subset=['Risk Tier']) \
            .applymap(color_prediction, subset=['Predicted No-Show']) \
            .format({'Risk Score (%)': '{:.2f}%'})

        st.dataframe(styled, use_container_width=True, height=480)

        # Summary stats
        st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("**Risk Score Stats**")
            st.dataframe(
                result_df['Risk Score (%)'].describe().round(2).rename('Value'),
                use_container_width=True
            )
        with s2:
            st.markdown("**Risk Tier Counts**")
            tier_summary = result_df['Risk Tier'].value_counts().reset_index()
            tier_summary.columns = ['Risk Tier','Count']
            tier_summary['Percentage'] = (tier_summary['Count']/total*100).round(1).astype(str) + '%'
            st.dataframe(tier_summary, use_container_width=True)
        with s3:
            st.markdown("**Waiting Band Distribution**")
            wband = result_df['Waiting Band'].value_counts().reset_index()
            wband.columns = ['Waiting Band','Count']
            st.dataframe(wband, use_container_width=True)

    # ── TAB 3: HIGH RISK PATIENTS ─────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">🔴 High Risk Patients — Immediate Action Required</div>', unsafe_allow_html=True)

        high_df = result_df[result_df['Risk Tier'] == 'High'].copy()
        high_df = high_df.sort_values('Risk Score (%)', ascending=False).reset_index(drop=True)

        st.error(f"⚠️ **{len(high_df):,} patients** are at HIGH risk of missing their appointment (risk score ≥ {high_thresh*100:.0f}%)")

        # High risk table columns
        hr_cols = []
        for c in ['PatientId','AppointmentDay','Age','Neighbourhood',
                  'Lead Time (Days)','Waiting Band','Risk Score (%)','Risk Tier','Predicted No-Show']:
            if c in high_df.columns:
                hr_cols.append(c)

        styled_hr = high_df[hr_cols].style \
            .applymap(color_tier, subset=['Risk Tier']) \
            .applymap(color_prediction, subset=['Predicted No-Show']) \
            .format({'Risk Score (%)': '{:.2f}%'}) \
            .background_gradient(subset=['Risk Score (%)'], cmap='Reds')

        st.dataframe(styled_hr, use_container_width=True, height=420)

        # Risk score histogram — high risk only
        st.markdown('<div class="section-header">High Risk Score Distribution</div>', unsafe_allow_html=True)
        fig_hr = px.histogram(
            high_df, x='Risk Score (%)', nbins=20,
            color_discrete_sequence=['#D85A30'],
            title='Distribution of Risk Scores Among High Risk Patients'
        )
        fig_hr.update_layout(bargap=0.05)
        st.plotly_chart(fig_hr, use_container_width=True)

    # ── TAB 4: SMS NOTIFICATIONS ──────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">📱 SMS Reminder Notifications</div>', unsafe_allow_html=True)
        st.markdown("""
        Send automated SMS reminders to high-risk patients to reduce no-show rates.
        This panel shows a **simulated SMS system** — integrate with **Twilio / AWS SNS / MSG91**
        for real SMS delivery.
        """)

        high_sms_df = result_df[result_df['Risk Tier'] == 'High'].copy()
        high_sms_df = high_sms_df.sort_values('Risk Score (%)', ascending=False).reset_index(drop=True)

        # SMS Control Panel
        st.markdown('<div class="section-header">SMS Control Panel</div>', unsafe_allow_html=True)
        ctrl1, ctrl2, ctrl3 = st.columns(3)

        with ctrl1:
            st.metric("Total High Risk", f"{len(high_sms_df):,}")
        with ctrl2:
            send_limit = min(sms_limit, len(high_sms_df))
            st.metric("SMS to Send", f"{send_limit}")
        with ctrl3:
            st.metric("Estimated Cost (₹0.15/SMS)", f"₹{send_limit * 0.15:.2f}")

        # SMS template editor
        st.markdown('<div class="section-header">SMS Template</div>', unsafe_allow_html=True)
        sms_template = st.text_area(
            "Customize your SMS template",
            value=(
                "Dear Patient, this is a reminder for your medical appointment on {AppointmentDay}. "
                "Our system has flagged your appointment as high risk. "
                "Please confirm attendance by replying YES or call 1800-CLINIC to reschedule. "
                "Your Risk Score: {Risk Score (%):.1f}%"
            ),
            height=100
        )

        # Initialize SMS sent state
        if 'sms_sent_ids' not in st.session_state:
            st.session_state['sms_sent_ids'] = set()

        # Buttons
        b1, b2, b3 = st.columns(3)
        with b1:
            send_all = st.button(f"📤 Send SMS to Top {send_limit} High Risk Patients",
                                 type="primary", use_container_width=True)
        with b2:
            clear_btn = st.button("🗑️ Clear Sent Log", use_container_width=True)
        with b3:
            preview_btn = st.button("👁️ Preview Sample SMS", use_container_width=True)

        if clear_btn:
            st.session_state['sms_sent_ids'] = set()
            st.success("Sent log cleared.")

        if preview_btn and len(high_sms_df) > 0:
            sample = high_sms_df.iloc[0]
            st.info(f"**Sample SMS Preview:**\n\n{generate_sms(sample)}")

        # Send SMS
        if send_all and len(high_sms_df) > 0:
            to_send = high_sms_df.head(send_limit)
            progress = st.progress(0, text="Sending SMS notifications...")
            sent_count = 0

            for i, (idx, row) in enumerate(to_send.iterrows()):
                pid = str(row.get('PatientId', idx))
                st.session_state['sms_sent_ids'].add(pid)
                sent_count += 1
                progress.progress((i + 1) / len(to_send),
                                  text=f"Sending {i+1}/{len(to_send)} — Patient ID: ...{pid[-4:]}")
                time.sleep(0.05)

            progress.empty()
            st.success(f"✅ Successfully sent {sent_count} SMS reminders!")

        # Display SMS log
        st.markdown('<div class="section-header">SMS Notification Log</div>', unsafe_allow_html=True)

        if len(high_sms_df) == 0:
            st.info("No high-risk patients found. Adjust the risk threshold in the sidebar.")
        else:
            display_sms = high_sms_df.head(send_limit)

            for _, row in display_sms.iterrows():
                pid = str(row.get('PatientId', ''))
                is_sent = pid in st.session_state['sms_sent_ids'] or sms_auto_send

                # Individual send checkbox per patient
                col_chk, col_card = st.columns([0.08, 0.92])

                with col_chk:
                    individual_send = st.checkbox("", key=f"sms_{pid}_{_}",
                                                   value=is_sent)
                    if individual_send:
                        st.session_state['sms_sent_ids'].add(pid)

                with col_card:
                    status = "✅ SENT" if (pid in st.session_state['sms_sent_ids']) else "⏳ PENDING"
                    card_class = "sms-card-sent" if "SENT" in status else "sms-card"
                    appt_date = str(row.get('AppointmentDay', 'N/A'))[:10]
                    neighbourhood = row.get('Neighbourhood', 'N/A')
                    risk_pct = row['Risk Score (%)']
                    lead_time = row.get('Lead Time (Days)', 'N/A')

                    st.markdown(f"""
                    <div class="{card_class}">
                        <b>Patient ID:</b> ...{pid[-6:]} &nbsp;|&nbsp;
                        <b>Appointment:</b> {appt_date} &nbsp;|&nbsp;
                        <b>Neighbourhood:</b> {neighbourhood} &nbsp;|&nbsp;
                        <b>Wait Days:</b> {lead_time} &nbsp;|&nbsp;
                        <b>Risk Score:</b> <span style="color:#C0392B;font-weight:bold">{risk_pct:.1f}%</span>
                        &nbsp;|&nbsp; <b>Status:</b> {status}
                        <br><br>
                        <i style="color:#555">📨 {generate_sms(row)}</i>
                    </div>
                    """, unsafe_allow_html=True)

            # SMS summary table
            st.markdown('<div class="section-header">SMS Summary</div>', unsafe_allow_html=True)
            sms_summary = display_sms[['Risk Score (%)','Lead Time (Days)','Waiting Band']].copy()
            if 'PatientId'      in display_sms.columns: sms_summary.insert(0,'PatientId',display_sms['PatientId'])
            if 'AppointmentDay' in display_sms.columns: sms_summary['Appt Date'] = display_sms['AppointmentDay'].astype(str).str[:10]
            if 'Neighbourhood'  in display_sms.columns: sms_summary['Neighbourhood'] = display_sms['Neighbourhood']

            sms_summary['SMS Status'] = sms_summary.get('PatientId','').apply(
                lambda p: '✅ Sent' if str(p) in st.session_state['sms_sent_ids'] else '⏳ Pending'
            )
            st.dataframe(sms_summary.reset_index(drop=True), use_container_width=True)

        # Twilio integration guide
        with st.expander("🔧 How to Enable Real SMS (Twilio Integration)"):
            st.code("""
# Install Twilio: pip install twilio

from twilio.rest import Client

account_sid = 'YOUR_TWILIO_ACCOUNT_SID'
auth_token  = 'YOUR_TWILIO_AUTH_TOKEN'
client      = Client(account_sid, auth_token)

def send_real_sms(to_number, message):
    message = client.messages.create(
        body    = message,
        from_   = '+1XXXXXXXXXX',  # Your Twilio number
        to      = to_number        # Patient's phone number
    )
    return message.sid

# Usage — add phone number column to your CSV:
for _, row in high_risk_df.iterrows():
    send_real_sms(row['phone_number'], generate_sms(row))
            """, language='python')

        with st.expander("🔧 AWS SNS Integration"):
            st.code("""
import boto3

sns = boto3.client('sns', region_name='ap-south-1')

def send_sns_sms(phone_number, message):
    response = sns.publish(
        PhoneNumber = phone_number,   # '+919876543210'
        Message     = message,
        MessageAttributes={
            'AWS.SNS.SMS.SenderID': {
                'DataType': 'String',
                'StringValue': 'ClinicApp'
            },
            'AWS.SNS.SMS.SMSType': {
                'DataType': 'String',
                'StringValue': 'Transactional'
            }
        }
    )
    return response['MessageId']
            """, language='python')

    # ── TAB 5: DOWNLOAD ───────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">⬇️ Download Prediction Results</div>', unsafe_allow_html=True)

        # Prepare clean download dataframe
        dl_cols = []
        for c in ['PatientId','AppointmentID','AppointmentDay','ScheduledDay',
                  'Age','Gender','Neighbourhood','Scholarship',
                  'Hypertension','Diabetes','Alcoholism','SMS_received',
                  'Lead Time (Days)','Waiting Band',
                  'Risk Score (%)','Risk Score Raw','Risk Tier','Predicted No-Show']:
            if c in result_df.columns:
                dl_cols.append(c)
        if 'Actual No-Show' in result_df.columns:
            dl_cols.append('Actual No-Show')

        full_dl_df   = result_df[dl_cols].sort_values('Risk Score (%)', ascending=False)
        high_dl_df   = full_dl_df[full_dl_df['Risk Tier'] == 'High']
        med_dl_df    = full_dl_df[full_dl_df['Risk Tier'] == 'Medium']
        sms_log_df   = high_dl_df.copy()
        sms_log_df['SMS Status'] = sms_log_df['PatientId'].apply(
            lambda p: 'Sent' if str(p) in st.session_state.get('sms_sent_ids', set()) else 'Pending'
        )

        today = datetime.now().strftime('%Y%m%d_%H%M')

        d1, d2 = st.columns(2)

        with d1:
            st.markdown("#### 📄 Full Prediction File")
            st.markdown(f"All **{len(full_dl_df):,}** records with risk scores, tiers, and predicted no-show status.")
            csv_full = full_dl_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download All Predictions CSV",
                data=csv_full,
                file_name=f"all_predictions_{today}.csv",
                mime='text/csv',
                use_container_width=True,
                type="primary"
            )

            st.markdown("---")
            st.markdown("#### 🔴 High Risk Patients Only")
            st.markdown(f"**{len(high_dl_df):,}** high-risk patients for immediate clinic action.")
            csv_high = high_dl_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download High Risk CSV",
                data=csv_high,
                file_name=f"high_risk_patients_{today}.csv",
                mime='text/csv',
                use_container_width=True
            )

        with d2:
            st.markdown("#### 🟡 Medium Risk Patients")
            st.markdown(f"**{len(med_dl_df):,}** medium-risk patients for monitoring.")
            csv_med = med_dl_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Medium Risk CSV",
                data=csv_med,
                file_name=f"medium_risk_patients_{today}.csv",
                mime='text/csv',
                use_container_width=True
            )

            st.markdown("---")
            st.markdown("#### 📱 SMS Notification Log")
            st.markdown(f"High-risk patients with SMS send status (Sent / Pending).")
            csv_sms = sms_log_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download SMS Log CSV",
                data=csv_sms,
                file_name=f"sms_log_{today}.csv",
                mime='text/csv',
                use_container_width=True
            )

        # Preview of download file
        st.markdown('<div class="section-header">Preview: All Predictions File</div>', unsafe_allow_html=True)
        st.markdown(f"Showing first 20 rows of {len(full_dl_df):,} total records")
        st.dataframe(
            full_dl_df.head(20).style
            .applymap(color_tier, subset=['Risk Tier'])
            .applymap(color_prediction, subset=['Predicted No-Show'])
            .format({'Risk Score (%)': '{:.2f}%', 'Risk Score Raw': '{:.4f}'}),
            use_container_width=True,
            height=400
        )

        # Model performance footer
        st.markdown('<div class="section-header">Model Performance Reference</div>', unsafe_allow_html=True)
        pm1, pm2, pm3, pm4, pm5 = st.columns(5)
        pm1.metric("AUC-ROC",   metadata['auc_roc'],   help="Area under ROC curve — how well model separates classes")
        pm2.metric("F1-Score",  metadata['f1_score'],  help="Balance of precision and recall")
        pm3.metric("Precision", metadata['precision'], help="Of predicted no-shows, how many were actually no-shows")
        pm4.metric("Recall",    metadata['recall'],    help="Of actual no-shows, how many did the model catch")
        pm5.metric("Accuracy",  metadata['accuracy'],  help="Overall correct predictions on test set")

except Exception as e:
    st.error(f"❌ Error processing file: {e}")
    st.exception(e)
    st.info("Please make sure your CSV has the required columns: ScheduledDay, AppointmentDay, Age, Gender, Neighbourhood")
