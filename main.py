import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import yfinance as yf
from datetime import datetime
import os

# Step 1: Load the stock market data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

# Step 2: Compute technical indicators
def compute_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'] = compute_macd(df['Close'])
    df = df.dropna()
    return df

# Step 2a: Compute RSI (Relative Strength Index)
def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Step 2b: Compute MACD (Moving Average Convergence Divergence)
def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

# Step 3: Preprocess the data
def preprocess_data(df):
    df = df[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Volume']]
    df = df.dropna()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    return scaled_data, scaler

# Step 4: Create the dataset
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, 0])  # Predicting 'Close' price
    X = np.array(X)
    y = np.array(y)
    return X, y

# Step 5: Build the LSTM model
def build_model(input_shape, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# Step 6: Train the model with hyperparameter tuning
def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    return model, history

# Step 7: Evaluate the model
def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    return predictions, mse, r2

# Step 8: Save the model
def save_model(model, filename='stock_predictor_model.h5'):
    model.save(filename)
    print(f"Model saved as {filename}")

# Step 9: Load the model
def load_saved_model(filename='stock_predictor_model.h5'):
    if os.path.exists(filename):
        model = load_model(filename)
        print(f"Model loaded from {filename}")
        return model
    else:
        print("Model file not found!")
        return None

# Step 10: Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

# Step 11: Visualize the predictions
def plot_predictions(df, predictions, scaler, time_step):
    df_pred = df[-len(predictions):].copy()
    df_pred['Predictions'] = predictions

    plt.figure(figsize=(16, 8))
    plt.title('Stock Price Prediction with Additional Features')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(df_pred.index, scaler.inverse_transform(df_pred[['Close']]), label='Actual Price')
    plt.plot(df_pred.index, df_pred['Predictions'], label='Predicted Price')
    plt.legend(loc='lower right')
    plt.show()

# Step 12: Main function to run the entire process
def main():
    # Define the stock ticker, start and end dates
    ticker = 'AAPL'
    start = '2015-01-01'
    end = datetime.today().strftime('%Y-%m-%d')

    # Load and preprocess the data
    df = load_data(ticker, start, end)
    df = compute_technical_indicators(df)
    scaled_data, scaler = preprocess_data(df)

    # Create datasets
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)

    # Split the data into training and testing sets
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build, train and evaluate the model
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model, history = train_model(model, X_train, y_train, epochs=10, batch_size=32)

    # Plot training history
    plot_training_history(history)

    # Evaluate the model
    predictions, mse, r2 = evaluate_model(model, X_test, y_test, scaler)

    # Visualize the predictions
    plot_predictions(df, predictions, scaler, time_step)

    # Save the trained model
    save_model(model)

if __name__ == "__main__":
    main()
