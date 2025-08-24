# importing libraries 
import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

@st.cache_data  # The data is cached and not re-downloaded unless the inputs change
def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            st.error("No data found for the given ticker or date range.") # Checks is the data founded. If not,shows an ERROR and exits
            return None
        return stock_data
    except Exception as e:   # Handels any download errors
        st.error(f"Error fetching data: {str(e)}")
        return None

def prepare_prediction_data(stock_data, scaler, lookback=100):  # Prepares the stock data to feed the ML model
    x_test = stock_data[['Close']].values  # Takes the "Close" coloumn of the stock data
    scaled_data = scaler.fit_transform(x_test)  # scales closing price into range of 0 to 1
    
    x_data, y_data = [], []
    for i in range(lookback, len(scaled_data)):  # Creates sequence of 100 past days in x_data (input data) and the next day as y_data (output data)
        x_data.append(scaled_data[i - lookback : i])
        y_data.append(scaled_data[i])
    
    if len(x_data) == 0:  # If data is too short
        st.error("Insufficient data for prediction after preprocessing.")
        return None, None
    
    return np.array(x_data), np.array(y_data) # Returns numpy arrays 

def plot_graph(data_series, full_data, label="Series", extra_data=None, extra_label=None):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(full_data.index, full_data['Close'], label="Close Price", color="blue", alpha=0.6) 
    plt.plot(data_series.index, data_series, label=label, color="orange", linewidth=2)
    
    if extra_data is not None and extra_label: # Optionally adds extra graph, for example: moving average of 250 days
        plt.plot(extra_data.index, extra_data, label=extra_label, color="green", linestyle="--")
    
    # Graph representation
    plt.title(f"{label} vs Close Price", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (USD)", fontsize=12)
    plt.legend()
    plt.grid(True)
    return fig

# Sets the layout of the app
st.set_page_config(page_title="Stock Market Predictor", layout="wide", initial_sidebar_state="expanded")

# Injects custom HTML to style headers, buttons and background
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stTextInput>div>input {border-radius: 5px;}
    h1, h2, h3 {color: #2c3e50;}
    .stMetric {background-color: #333533; border-radius: 5px; padding: 10px;}
    </style>
""", unsafe_allow_html=True)


st.sidebar.title("📊 Stock Settings")  # Side bar allows user to change stock data and end date
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TSLA, GOOG)", value="GOOG").upper()
end_date = st.sidebar.date_input("Select End Date", value=datetime.today())
start_date = end_date - timedelta(days=20*365)  # Sets start date as the past 20 years from the end date

# If predictions is not already a part of the coloumn, then it is added
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None


try:  
    model = load_model("Latest_Stock_Price_Model.keras")  # Loads the ML model
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()  # Handles errors while loading model and stops the app execution


st.title("📈 Stock Market Analysis & Predictions")
st.markdown(f"Analyze historical data and predict future prices for *{ticker}*. Use the sidebar to customize settings.")

with st.spinner("Fetching stock data..."):
    stock_data = fetch_stock_data(ticker, start_date, end_date) # Downloads stock data from yfinance

if stock_data is None: 
    st.stop()  # If stock data is not found, it stops the app execution

tab1, tab2, tab3 = st.tabs(["📊 Historical Data", "📉 Analysis", "🤖 Predictions"]) 


with tab1:
    st.subheader(f"{ticker} Historical Data")
    st.dataframe(stock_data.tail(500), height=400)  # Show recent data


with tab2:
    st.subheader(f"{ticker} Market Analysis")
   
    if stock_data.empty or not all(col in stock_data.columns for col in ['High', 'Low', 'Close']): # Excepts errors like inconsistency in data 
        st.error("Stock data is empty or missing required columns (High, Low, Close).")
        st.stop()

    col1, col2, col3 = st.columns(3)
    try:
        with col1:
            high_value = float(stock_data['High'].iloc[-1]) # Display's the last value of the high coloumn
            st.metric("Current High", f"${high_value:.2f}")
        with col2:
            low_value = float(stock_data['Low'].iloc[-1])
            st.metric("Current Low", f"${low_value:.2f}")
        with col3:
            close_value = float(stock_data['Close'].iloc[-1])
            st.metric("Current Close", f"${close_value:.2f}")
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")
        st.stop()
        
    # Calculation Moving Average (MA) for different windows 
    stock_data['MA_100'] = stock_data['Close'].rolling(window=100).mean()  
    stock_data['MA_200'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['MA_250'] = stock_data['Close'].rolling(window=250).mean()


    # Plotting the graphs for the moving average
    st.subheader("📉 Moving Averages")
    st.pyplot(plot_graph(stock_data['MA_100'], stock_data, "100-Day MA"))
    st.pyplot(plot_graph(stock_data['MA_200'], stock_data, "200-Day MA"))
    st.pyplot(plot_graph(stock_data['MA_250'], stock_data, "250-Day MA"))
    st.pyplot(plot_graph(stock_data['MA_100'], stock_data, "100-Day MA", stock_data['MA_250'], "250-Day MA"))

with tab3:
    st.subheader(f"{ticker} Price Predictions")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_data, y_data = prepare_prediction_data(stock_data, scaler)  # Prepares the data for ML model
    
    if x_data is None or y_data is None:
        st.stop()

    with st.spinner("Generating predictions..."):
        predictions = model.predict(x_data, verbose=0) 
        inverse_predictions = scaler.inverse_transform(predictions)  # Changing the predictions back to normal
        inverse_y_test = scaler.inverse_transform(y_data)

    predicted_close = float(inverse_predictions[-1][0]) 
    last_close_price = float(stock_data['Close'].iloc[-1])  
    predicted_open = predicted_close # For simplicity
    predicted_high = predicted_close * 1.05  # 5% Increase
    predicted_low = predicted_close * 0.95  # 5% decrease

    col1, col2, col3, col4 = st.columns(4) # Displaying data 
    with col1:
        st.metric("Predicted Open", f"${predicted_open:.2f}")
    with col2:
        st.metric("Predicted High", f"${predicted_high:.2f}")
    with col3:
        st.metric("Predicted Low", f"${predicted_low:.2f}")
    with col4:
        st.metric("Predicted Close", f"${predicted_close:.2f}")

    st.subheader("Trend Outlook")  # Predicting if the stock will go up or down
    if predicted_close > last_close_price:
        st.success("🚀 The stock is predicted to go *UP*!")
    else:
        st.error("📉 The stock is predicted to go *DOWN*!")

    mse = mean_squared_error(inverse_y_test, inverse_predictions)  # Calculating the error rate 
    target_range = np.max(inverse_y_test) - np.min(inverse_predictions)
    percentage_accuracy = (1.0 - (mse / target_range)) * 100  # Converting to model accuracy percentage
    st.metric("Model Percentage Accuracy", f"{percentage_accuracy:.2f} %")

    st.subheader("Predictions vs Actual Prices")  # Plotting the graph for predictions v/s actual closing price
    fig = plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index[-len(inverse_predictions):], inverse_predictions, label="Predicted Close", color="red")
    plt.plot(stock_data.index[-len(inverse_y_test):], inverse_y_test, label="Actual Close", color="blue", alpha=0.6) # aplha = for opacity
    plt.title("Predicted vs Actual Closing Prices", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (USD)", fontsize=12)
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)