# --------------------------------------------------
# [4] Streamlit 예측 앱
# --------------------------------------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.title("☀️ Prophet Sunspot Forecast Dashboard")

@st.cache_data
def load_prophet_data():
    try:
        df = pd.read_csv("sunspots_for_prophet.csv")
    except Exception:
        try:
            df = pd.read_csv("data/sunspots.csv")
            df['YEAR'] = df['YEAR'].astype(int)
            df = df[(df['YEAR'] >= 1900) & (df['YEAR'] <= 2008)]
            df['ds'] = pd.to_datetime(df['YEAR'].astype(str), format='%Y')
            df = df[['ds', 'SUNACTIVITY']].rename(columns={'SUNACTIVITY': 'y'})
        except Exception:
            import statsmodels.api as sm
            df = sm.datasets.sunspots.load_pandas().data
            df['YEAR'] = df['YEAR'].astype(int)
            df = df[(df['YEAR'] >= 1900) & (df['YEAR'] <= 2008)]
            df['ds'] = pd.to_datetime(df['YEAR'].astype(str), format='%Y')
            df = df[['ds', 'SUNACTIVITY']].rename(columns={'SUNACTIVITY': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    return df

try:
    data = load_prophet_data()
    st.write("Prophet 예측 입력 데이터", data.head())

    periods = st.slider("예측 기간 (연 단위)", 10, 100, 50)

    m = Prophet(yearly_seasonality=False, changepoint_prior_scale=0.05)
    m.add_seasonality(name='sunspot_cycle', period=11, fourier_order=5)
    m.fit(data)

    try:
        future = m.make_future_dataframe(periods=periods, freq='YE')
    except Exception:
        future = m.make_future_dataframe(periods=periods, freq='Y')

    forecast = m.predict(future)

    fig1 = m.plot(forecast)
    st.pyplot(fig1)

    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)
except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
