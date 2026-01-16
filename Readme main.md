# 📈 Market Trend Analysis using LSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 📌 Project Overview

This repository contains a deep learning project focused on **Market Trend Analysis** (Financial Forecasting). The goal is to predict stock price movements using historical data to assist in investment decision-making.

The core of this project is the Jupyter Notebook (`Market_Trend_Analysis.ipynb`), which serves as the **main source of truth** for evaluation. It covers the end-to-end pipeline from data extraction to model evaluation.

---

## 📂 Repository Structure

```text
├── Market_Trend_Analysis.ipynb   # PRIMARY SUBMISSION: The complete project notebook
├── README.md                     # Project documentation and summary
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git configuration

```
## 1. Problem Definition & Objective
   
🎯 Selected Track

Market Trend Analysis (Financial Forecasting)

❓ Problem Statement
Financial markets are characterized by high volatility and non-linear patterns. Traditional statistical methods often fail to capture long-term temporal dependencies. This project aims to build an AI model to predict the Close Price of a specific stock (Apple Inc. - AAPL) using historical daily data.

🌍 Real-World Relevance
Accurate trend prediction helps investors (retail and institutional) in:

* Risk Management: Identifying potential downturns.
* Portfolio Optimization: Timing market entry and exit.
* Reducing Emotional Bias: Providing data-driven insights.

## 3. Data Understanding & Preparation

Source: Publicly available market data via the Yahoo Finance API (yfinance).
Dataset: Historical daily stock prices (Open, High, Low, Close, Volume) for Apple Inc. (AAPL) from 2015 to 2024.
Preprocessing:
Feature Selection: Focused on the 'Close' price.
Normalization: Applied MinMaxScaler to scale data between 0 and 1 for optimal LSTM convergence.
Sequence Generation: Created a rolling window of 60 days (using past 60 days to predict the 61st day).

## 5. Model & System Design

🧠 AI Technique
Deep Learning (Recurrent Neural Networks - RNN) using Long Short-Term Memory (LSTM) units.

🏗️ Architecture
Input Layer: Shape (Batch_Size, 60, 1) representing 60 days of historical prices.
LSTM Layer 1: 50 units, return_sequences=True. Captures sequential patterns.
LSTM Layer 2: 50 units, return_sequences=False. Aggregates temporal features.
Dense Layers:
Dense(25): Feature condensation.
Dense(1): Output layer (predicted normalized price).

💡 Justification
Standard Feed-Forward Networks treat inputs as independent. Stock prices are sequential time-series data where order matters. LSTMs are specifically designed to solve the Vanishing Gradient Problem and retain long-term memory, making them superior for financial forecasting.


## 6. Core Implementation
The implementation is contained entirely within Market_Trend_Analysis.ipynb.
Library Stack: yfinance, pandas, numpy, scikit-learn, tensorflow/keras, matplotlib.
Training: The model is trained using the Adam optimizer and Mean Squared Error (MSE) loss function over 20 epochs.
Inference: The system predicts the next day's price based on the test set and inversely scales the output to get the actual dollar value.

## 7. Evaluation & Analysis
📉 Metrics
We utilize Root Mean Squared Error (RMSE) to measure the standard deviation of the prediction errors.

📊 Performance
The model was tested on unseen data (the most recent 20% of the dataset).
Visual analysis (plotted in the notebook) shows that the predicted trend lines closely follow the actual market movement, validating the model's ability to capture directionality.

## 8. Ethical Considerations
⚠️ Not Financial Advice: This AI tool is for educational and research purposes only. It should not be the sole basis for real-money trading decisions.
Bias & Limitations: The model relies on historical patterns. It may not predict "Black Swan" events (e.g., global pandemics, sudden regulatory changes) that were not present in the training data.
Responsible AI: Users must be aware that all financial forecasting allows for probabilities, not certainties.

## 9. Conclusion & Future Scope
✅ Summary
We successfully implemented an LSTM-based pipeline that automates data fetching, preprocessing, and trend forecasting. The model demonstrates that Deep Learning can effectively extract patterns from noisy financial data.
🚀 Future Improvements
Sentiment Analysis: Integrate news headlines using NLP (BERT/RoBERTa) to factor in market sentiment.
Multivariate Analysis: Include Volume, RSI, and MACD indicators as input features.
Hyperparameter Tuning: Use GridSearch to optimize LSTM units and learning rates.
