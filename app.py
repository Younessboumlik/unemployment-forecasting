import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Prévision Chômage US", layout="wide")

DIR_MODELS = "trained_models_jupyter"
PROPHET_PATH = os.path.join(DIR_MODELS, "mon_modele_prophet_entraine.pkl")
ARIMA_PATH = os.path.join(DIR_MODELS, "modele_ARIMA.pkl")

def load_and_preprocess(data_path):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.upper()
    date_col = next((c for c in df if c in ("DATE", "OBSERVATION_DATE")), None)
    value_col = next((c for c in df if c in ("UNRATE", "VALUE")), None)
    if date_col is None or value_col is None:
        st.error("Colonnes de date ou de taux introuvables.")
        st.stop()
    df = df[[date_col, value_col]].dropna()
    df = df.rename(columns={date_col: 'ds', value_col: 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    return df.sort_values('ds')

DATA_FILE = 'UNRATE.csv'
if not os.path.exists(DATA_FILE):
    st.error(f"Fichier manquant: {DATA_FILE}")
    st.stop()

df = load_and_preprocess(DATA_FILE)
last_date = df['ds'].max()

if not os.path.exists(PROPHET_PATH) or not os.path.exists(ARIMA_PATH):
    st.error("Modèles pré-entraînés introuvables.")
    st.stop()

model_prophet = joblib.load(PROPHET_PATH)
model_arima = joblib.load(ARIMA_PATH)

st.title("Prévision du Taux de Chômage US")
params = st.sidebar
params.header("Paramètres de Prévision")
freq = params.selectbox("Unité de temps", ["Mois", "Années"], index=0)
n = params.number_input("Nombre à prévoir", min_value=1, max_value=120, value=12)
periods = n if freq == "Mois" else n * 12

if params.button("Lancer la prévision"):
    future = model_prophet.make_future_dataframe(periods=periods, freq='MS')
    forecast = model_prophet.predict(future)
    fut = forecast[forecast['ds'] > last_date]
    df_prop = fut[['ds','yhat','yhat_lower','yhat_upper']].rename(
        columns={'ds':'Date','yhat':'Prophet','yhat_lower':'Prophet_low','yhat_upper':'Prophet_high'}
    )

    arima_fc = model_arima.get_forecast(steps=periods)
    mean_pred = arima_fc.predicted_mean
    ci = arima_fc.conf_int(alpha=0.2)
    dates_ar = pd.date_range(
        start=(last_date + pd.DateOffset(months=1)),
        periods=periods,
        freq='MS'
    )
    df_ar = pd.DataFrame({
        'Date': dates_ar,
        'ARIMA': mean_pred.values,
        'ARIMA_low': ci.iloc[:,0].values,
        'ARIMA_high': ci.iloc[:,1].values
    })

    # Affichage tables
    st.subheader("Tableau des Prévisions")
    merged = pd.merge(df_prop, df_ar, on='Date')
    merged_display = merged.copy()
    merged_display['Date'] = merged_display['Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(merged_display)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Historique'))
    fig.add_trace(go.Scatter(x=merged['Date'], y=merged['Prophet'], mode='lines', name='Prophet'))
    fig.add_trace(go.Scatter(
        x=list(merged['Date']) + list(merged['Date'])[::-1],
        y=list(merged['Prophet_high']) + list(merged['Prophet_low'])[::-1],
        fill='toself', name='IC Prophet (80%)', hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(x=merged['Date'], y=merged['ARIMA'], mode='lines', name='ARIMA'))
    fig.add_trace(go.Scatter(
        x=list(merged['Date']) + list(merged['Date'])[::-1],
        y=list(merged['ARIMA_high']) + list(merged['ARIMA_low'])[::-1],
        fill='toself', name='IC ARIMA (80%)', hoverinfo='skip'
    ))
    fig.add_shape(
        dict(
            type='line', x0=last_date, x1=last_date,
            yref='paper', y0=0, y1=1,
            line=dict(color='gray', dash='dash')
        )
    )
    fig.add_annotation(
        dict(
            x=last_date, y=1.02, xref='x', yref='paper',
            text='Dernière donnée', showarrow=False, yanchor='bottom'
        )
    )
    fig.update_layout(
        title='Prévisions du Taux de Chômage US',
        xaxis_title='Date', yaxis_title='Taux (%)', hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
