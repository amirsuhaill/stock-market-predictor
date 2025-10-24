# 📈 Stock Market Predictor

A deep learning–based **Stock Price Prediction and Visualization Web App** built using **LSTM (Long Short-Term Memory)** networks to forecast future closing prices of stocks.
The project leverages **historical market data** from Yahoo Finance and provides **interactive visualizations** through a Streamlit-based web interface.

---

## 🚀 Features

- 🧠 **LSTM Deep Learning Model** trained on 10 years of stock data to predict future closing prices.
- 💹 **Interactive Streamlit Dashboard** to visualize real-time stock trends, predictions, and moving averages.
- 📊 **Technical Indicators:** MA50, MA100, and MA200 visual comparisons for trend analysis.
- ⚙️ **Automated Data Fetching** using `yfinance` API.
- 🔍 **Scalable Preprocessing Pipeline** using MinMax normalization and sequential windowing for time-series learning.

---

## 🧰 Tech Stack

| Category                          | Technologies              |
| --------------------------------- | ------------------------- |
| **Frontend**                | Streamlit                 |
| **Backend / ML**            | Python, Keras, TensorFlow |
| **Data Handling**           | Pandas, NumPy, yFinance   |
| **Visualization**           | Matplotlib                |
| **Scaling / Preprocessing** | Scikit-learn              |

---

## 📂 Project Structure


Stock-Market-Predictor/

├── app.py                # Streamlit web app for prediction and visualization

├── model.py              # LSTM model training and evaluation script

├── Stock Predictions Model.keras  # Saved trained LSTM model

├── requirements.txt      # Required Python dependencies

└── README.md             # Project documentation

<pre class="overflow-visible!" data-start="1834" data-end="2004"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"></div></div></pre>


---
## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/amirsuhaill/stock-market-predictor.git
cd stock-market-predictor
---
pip install -r requirements.txt

streamlit run app.py
