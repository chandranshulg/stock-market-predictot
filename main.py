import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import yfinance as yf
from datetime import datetime

# Step 1: Load the stock market data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

# Step 2: Preprocess the data
def preprocess_data(df):
    df = df[['Close']]
    df = df.dropna()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    return scaled_data, scaler

# Step 3: Create the dataset
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

# Step 4: Build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Step 5: Train the model
def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

# Step 6: Make predictions
def predict_stock(model, data, time_step):
    predictions = []
    for i in range(time_step, len(data)):
        x_test = data[i-time_step:i, 0]
        x_test = np.reshape(x_test, (1, x_test.shape[0], 1))
        predicted_price = model.predict(x_test)
        predictions.append(predicted_price[0, 0])
    return np.array(predictions)

# Step 7: Visualize the results
def plot_predictions(df, predictions, scaler):
    train = df[:int(len(df) * 0.7)]
    valid = df[int(len(df) * 0.7):]
    valid['Predictions'] = scaler.inverse_transform(predictions.reshape(-1, 1))

    plt.figure(figsize=(16, 8))
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

# Step 8: Main function to run the entire process
def main():
    # Define the stock ticker, start and end dates
    ticker = 'AAPL'
    start = '2015-01-01'
    end = datetime.today().strftime('%Y-%m-%d')

    # Load and preprocess the data
    df = load_data(ticker, start, end)
    scaled_data, scaler = preprocess_data(df)

    # Create datasets
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)

    # Split the data into training and testing sets
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build and train the model
    model = build_model((X_train.shape[1], 1))
    model = train_model(model, X_train, y_train, epochs=10, batch_size=32)

    # Predict the stock prices
    predictions = predict_stock(model, scaled_data[split:], time_step)

    # Visualize the predictions
    plot_predictions(df, predictions, scaler)

if __name__ == "__main__":
    main()
