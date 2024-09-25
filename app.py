import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from yahooquery import search
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime

# Function to download historical data for the ticker
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to prepare dataset for LSTM
def prepare_data(data, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])
        y.append(scaled_data[i, 3])  # 'Close' is the target
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Function to build and train LSTM model
def train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=250, verbose=0)
    return model

# Function to predict the closing price for a specified date
def predict_closing_price(model, scaler, ticker, seq_length):
    latest_data = yf.download(ticker, period='1d', interval='1m')
    latest_data = latest_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    
    scaled_latest_data = scaler.transform(latest_data)
    latest_sequence = scaled_latest_data[-seq_length:]
    
    # Ensure latest_sequence has the correct shape
    if latest_sequence.shape[0] != seq_length:
        raise ValueError(f"Expected sequence length of {seq_length}, but got {latest_sequence.shape[0]}")

    latest_sequence = np.reshape(latest_sequence, (1, seq_length, latest_sequence.shape[1]))
    
    predicted_price = model.predict(latest_sequence, verbose=0)
    predicted_price = scaler.inverse_transform([[0, 0, 0, predicted_price[0, 0], 0]])
    return predicted_price[0, 3]

# Main app function
def main():
    # Add CSS to set a background image
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url("C:\\Users\\Admin\\OneDrive\\Desktop\\app\\th.jpg);
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 100vh;
            color: white;  /* Optional: Change text color to make it visible on the background */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Stock Price Prediction App")
    
    # Input fields for stock name and symbol
    stock_name = st.text_input("Enter the stock name (e.g., 'HDFC Bank', 'Reliance Industries')", "")
    stock_symbol = st.text_input("Or enter the stock symbol (if known)", "")
    start_date = st.date_input("Start date", value=datetime(2020, 1, 1))
    prediction_date = st.date_input("Prediction date", value=datetime(2024, 8, 12))
    
    # Determine if the stock symbol is provided or needs to be fetched by name
    if stock_symbol:
        ticker = stock_symbol
        st.write(f"Using the provided stock symbol: {ticker}")
    elif stock_name:
        st.write(f"Fetching ticker for {stock_name}...")
        result = search(stock_name)
        if result['quotes']:
            ticker = result['quotes'][0]['symbol']
            st.write(f"The ticker symbol for {stock_name} is: {ticker}")
        else:
            st.write(f"No ticker symbol found for '{stock_name}'")
            return  # Exit if no symbol is found and none is provided
    else:
        st.write("Please enter either a stock name or symbol.")
        return
    
    if st.button("Predict Stock Price"):
        try:
            # Download data
            st.write("Downloading historical data...")
            data = download_data(ticker, start_date, prediction_date)  # End date set to prediction date
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Prepare data and train model
            seq_length = 60
            X_train, y_train, scaler = prepare_data(data.values, seq_length)
            model = train_model(X_train, y_train)
            
            # Predict the price
            st.write("Predicting the stock price...")
            predicted_price = predict_closing_price(model, scaler, ticker, seq_length)
            last_closing_price = data['Close'][-1:].values[0]
            st.write(f"Predicted Closing Price on {prediction_date}: {predicted_price:.2f}")
            st.write(f"Last Known Closing Price: {last_closing_price:.2f}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
