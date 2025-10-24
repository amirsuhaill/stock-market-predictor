import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.header('Stock Market Predictor')

# --- Inputs ---
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# --- Data ---
data = yf.download(stock, start=start, end=end)
if data is None or data.empty:
    st.error("No data downloaded. Check the symbol or your internet.")
    st.stop()

st.subheader('Stock Data')
st.write(data)

# --- Train/Test split (use only Close) ---
close = data[['Close']].copy()
split = int(len(close) * 0.80)
data_train = close.iloc[:split]
data_test = close.iloc[split:]

# --- Scaling: fit on TRAIN only, transform both ---
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(data_train)
# prepend last 100 train points for test windowing
past_100 = data_train.tail(100)
test_concat = pd.concat([past_100, data_test], ignore_index=True)
test_scaled = scaler.transform(test_concat)

# --- Moving averages ---
st.subheader('Price vs MA50')
ma_50 = close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50, label='MA50')
plt.plot(close, label='Close')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100 = close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50, label='MA50')
plt.plot(ma_100, label='MA100')
plt.plot(close, label='Close')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200 = close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100, label='MA100')
plt.plot(ma_200, label='MA200')
plt.plot(close, label='Close')
plt.legend()
st.pyplot(fig3)

# --- Build test windows ---
x, y = [], []
for i in range(100, test_scaled.shape[0]):
    x.append(test_scaled[i-100:i])      # shape (100, 1)
    y.append(test_scaled[i, 0])         # scalar
x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# ensure 3D for LSTM/CNN models: (samples, timesteps, features)
if x.ndim == 2:
    x = x.reshape((x.shape[0], x.shape[1], 1))

# --- Load model ---
model = load_model('Stock Predictions Model.keras')

# --- Predict and inverse-scale ---
pred_scaled = model.predict(x, verbose=0)
# pred_scaled could be shape (n,1) -> force 2D for inverse_transform
pred_inv = scaler.inverse_transform(np.asarray(pred_scaled).reshape(-1, 1)).ravel()
y_inv = scaler.inverse_transform(y.reshape(-1, 1)).ravel()

# --- Plot ---
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y_inv, label='Original Price', linewidth=1.5)
plt.plot(pred_inv, label='Predicted Price', linewidth=1.5)
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)