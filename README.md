# Stock Price Prediction 

This project involves building a stock price prediction model using Long Short-Term Memory (LSTM) networks. The model utilizes technical indicators like SMA, RSI, and MACD, to predict future stock prices.

## Features

- **Data Collection**: Fetches historical stock data from Yahoo Finance.
- **Technical Indicators**: Computes Simple Moving Average (SMA), Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD).
- **Data Preprocessing**: Scales the data using Min-Max scaling.
- **Model Building**: Constructs an LSTM model for predicting stock prices.
- **Model Training**: Trains the model with hyperparameter tuning.
- **Evaluation**: Evaluates model performance using Mean Squared Error (MSE) and R-squared (R²) metrics.
- **Visualization**: Plots training history and predictions.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow
- yfinance

## Installation

1. **Clone the Repository**:

     ```bash
    git clone https://github.com/chandranshulg/stock-market-predictot.git
    cd /stock-market-predictot

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   
3. **Create requirements.txt**:

   ```bash
   pip freeze > requirements.txt

**You can install the necessary packages using pip**:

    ```bash
    pip install pandas numpy matplotlib scikit-learn tensorflow yfinance


## Usage

1. **Update the Ticker Symbol**: Modify the ticker variable in the main() function to the stock symbol you wish to analyze.
2. **Run the Script**: Execute the script to perform data loading, preprocessing, model training, evaluation, and saving.  
   ```bash
   main.py
