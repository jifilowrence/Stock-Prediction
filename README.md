Amazon Stock Price Prediction

Overview

The Amazon Stock Price Prediction project uses Machine Learning to analyze historical Amazon (AMZN) stock data and predict the next available trading day's closing price. The project uses a Random Forest Regressor model and a Streamlit dashboard to present predictions, historical price trends, and model evaluation metrics.

The goal is to explore how historical stock prices and technical indicators can be used to estimate future prices.

Features

Historical Data Analysis: Load and analyze Amazon stock price data from a CSV file.

Data Preprocessing: Clean the dataset, handle missing values, and prepare features for training.

Feature Engineering: Generate lag features and moving averages.

Machine Learning: Train a Random Forest Regressor to predict the next trading day's price change.

Model Evaluation: Evaluate predictions using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

Interactive Dashboard: Display stock information, historical price charts, model performance, and predicted prices using Streamlit.

Actual vs. Predicted Visualization: Compare model predictions with actual historical closing prices.

Technologies Used

Python — Programming language

Pandas — Data manipulation and preprocessing

NumPy — Numerical computations

Matplotlib — Data visualization

Scikit-learn — Machine Learning and model evaluation

Streamlit — Interactive web dashboard

Google Colab — Model development and experimentation

Dataset

The project uses historical Amazon stock data from a CSV file named Amazon stock Data.csv.

The dataset is expected to contain stock market information such as:

Date

Open price

High price

Low price

Closing price

Trading volume

The exact column names depend on the dataset. The application should be configured to match the CSV's column names and format.

Machine Learning Approach

1. Data Preprocessing

The historical data is loaded from the CSV file, date values are converted to datetime format, numeric columns are cleaned, and records are sorted chronologically.

2. Feature Engineering

The model uses historical closing prices and derived features:

lag_1 — Previous trading day's closing price

lag_2 — Closing price two trading days earlier

lag_5 — Closing price five trading days earlier

ma_5 — Five-day moving average

ma_20 — Twenty-day moving average

3. Model Training

A Random Forest Regressor is trained to estimate the change between the current closing price and the next available trading day's closing price.

The dataset is split chronologically into training and testing sets to evaluate the model on later historical observations.

4. Model Evaluation

The model is evaluated using:

Mean Absolute Error (MAE): Measures the average absolute difference between actual and predicted prices.

Root Mean Squared Error (RMSE): Measures prediction error while penalizing larger errors more heavily.

Lower values generally indicate more accurate predictions. Model performance should also be compared with a simple baseline that predicts no change in the closing price.
