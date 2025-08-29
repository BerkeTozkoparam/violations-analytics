# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import plotly.graph_objs as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Load dataset ---
df = pd.read_csv("globalterrorismdb_0718dist.csv", encoding='latin1', low_memory=False)

# --- 2. Create date column ---
df['day'] = df['iday'].replace(0, 1)
df['month'] = df['imonth'].replace(0, 1)
df['date'] = pd.to_datetime(dict(year=df['iyear'], month=df['month'], day=df['day']))

# --- 3. Daily violations count ---
df_daily = df.groupby('date').size().reset_index(name='violations')

# --- 4. Lag and rolling features ---
df_daily['lag_1'] = df_daily['violations'].shift(1).fillna(0)
df_daily['lag_7'] = df_daily['violations'].rolling(7).sum().fillna(0)
df_daily['lag_30'] = df_daily['violations'].rolling(30).sum().fillna(0)

# --- 5. Features & target ---
X = df_daily[['lag_1','lag_7','lag_30']]
y = df_daily['violations']

# --- 6. Sequence-aware train/test split ---
split_idx = int(len(df_daily)*0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# --- 7. CatBoost Regressor ---
cat_model = CatBoostRegressor(
    iterations=1000,
    depth=5,
    learning_rate=0.05,
    eval_metric='RMSE',
    random_seed=42,
    early_stopping_rounds=20,
    verbose=50
)
cat_model.fit(X_train, y_train, eval_set=(X_test, y_test))

# --- 8. Predictions ---
y_pred = cat_model.predict(X_test)
df_daily.loc[X_test.index, 'pred_violations'] = y_pred

# --- 9. Risk level categories ---
def risk_category(pred):
    if pred < 3:
        return "Low"
    elif pred < 7:
        return "Medium"
    else:
        return "High"

df_daily.loc[X_test.index, 'risk_level_cat'] = df_daily.loc[X_test.index, 'pred_violations'].apply(risk_category)

# --- 10. Daily change & pct change for trend ---
df_daily['diff_violations'] = df_daily['violations'].diff().fillna(0)
df_daily['pct_change'] = df_daily['violations'].pct_change().fillna(0) * 100

# --- 11. Model performance ---
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# --- 12. Streamlit Dashboard ---
st.set_page_config(page_title="Military Threat Dashboard", layout="wide")
st.title("⚠️ Military Threat Prediction Dashboard")
st.markdown(f"**RMSE:** {rmse:.2f} | **R²:** {r2:.2f}")

# --- Sidebar Log / Notes & Metrics ---
st.sidebar.title("📜 Project Log / Notes")
st.sidebar.write("📅 First demo date: 2025-08-29")
max_idx = df_daily['pred_violations'].idxmax()
st.sidebar.write(f"🔴 Day with highest predicted risk: {df_daily['date'].iloc[max_idx].date()}")
st.sidebar.write("💡 Developed for ML and interactive dashboard experience.")
st.sidebar.subheader("📊 Recent Metrics")
st.sidebar.write("High Risk Days (last 7 days):", df_daily['risk_level_cat'][-7:].value_counts().get('High',0))
st.sidebar.write("Maximum Predicted Violations:", df_daily['pred_violations'].max())
st.sidebar.write("Average Predicted Violations:", df_daily['pred_violations'].mean().round(2))
st.sidebar.write("📈 Avg daily % change in violations:", f"{df_daily['pct_change'].mean():.2f}%")

# --- Risk Summary Metric ---
st.subheader("🛰️ Risk Summary")
high_risk_days = (df_daily['risk_level_cat']=='High').sum()
st.metric("High Risk Days (total)", high_risk_days)

# --- Time Series Panel with trend ---
fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(
    x=df_daily['date'], y=df_daily['violations'],
    mode='lines+markers', name='Actual', line=dict(color='white')
))
fig_ts.add_trace(go.Scatter(
    x=df_daily['date'], y=df_daily['pred_violations'],
    mode='lines+markers', name='Predicted', line=dict(color='green')
))
fig_ts.add_trace(go.Scatter(
    x=df_daily['date'][df_daily['risk_level_cat']=='High'],
    y=df_daily['pred_violations'][df_daily['risk_level_cat']=='High'],
    mode='markers', name='High Risk', marker=dict(color='red', size=8)
))
fig_ts.add_trace(go.Scatter(
    x=df_daily['date'][df_daily['diff_violations']>0],
    y=df_daily['violations'][df_daily['diff_violations']>0],
    mode='markers', name='Increasing Trend', marker=dict(color='lime', size=6, symbol='triangle-up')
))
fig_ts.add_trace(go.Scatter(
    x=df_daily['date'][df_daily['diff_violations']<0],
    y=df_daily['violations'][df_daily['diff_violations']<0],
    mode='markers', name='Decreasing Trend', marker=dict(color='orange', size=6, symbol='triangle-down')
))
fig_ts.update_layout(
    paper_bgcolor='black', plot_bgcolor='black', font_color='white',
    title="Daily Violations, Predicted Risk & Trend",
    xaxis_title="Date", yaxis_title="Number of Events"
)
st.plotly_chart(fig_ts, use_container_width=True)

# --- Country Daily Trend & Risk Map ---
df_country_daily = df.groupby(['date','country_txt']).size().reset_index(name='violations')
df_country_daily['diff_violations'] = df_country_daily.groupby('country_txt')['violations'].diff().fillna(0)
latest_date = df_country_daily['date'].max()
df_latest = df_country_daily[df_country_daily['date']==latest_date].copy()
df_latest['risk_level_cat'] = df_latest['violations'].apply(lambda x: 'Low' if x<3 else ('Medium' if x<7 else 'High'))
df_latest['trend_cat'] = df_latest['diff_violations'].apply(lambda x: 'Increasing' if x>0 else ('Decreasing' if x<0 else 'Stable'))

fig_map = px.choropleth(
    df_latest, locations='country_txt', locationmode='country names',
    color='trend_cat', hover_name='country_txt', hover_data=['violations','risk_level_cat'],
    color_discrete_map={'Increasing':'lime','Decreasing':'orange','Stable':'gray'},
    title=f'⚠️ Risk & Trend Map - {latest_date.date()}'
)
fig_map.update_layout(paper_bgcolor='black', plot_bgcolor='black', font_color='white')
st.subheader("🌐 Current Risk & Trend Map")
st.plotly_chart(fig_map, use_container_width=True)

# --- Table Panel ---
st.subheader("📊 Predictions & Risk Table")
st.dataframe(df_daily[['date','violations','pred_violations','risk_level_cat']].tail(50))
