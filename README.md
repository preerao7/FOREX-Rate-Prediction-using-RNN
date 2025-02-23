# FOREX Rate Predictions using GRU-LSTM Hybrid RNN Model

Foreign Exchange Currency Rate Prediction using a GRU-LSTM Hybrid Neural Nets

This project uses combination of Recurrent Neural Networks - GRU (Gated Reccurent Unit) and LSTM (Long Short Term Memory) to predict the FOREX Currency rates based on Historic OHLC (Open, High, Low, Close) data. 

GRU is known effeiciently and faster on time-series data and LSTM provides more accuracy on time-series data, so we combined both the models to take advantage of both the models improved effciency when worked together.

The dataset is preprocessed to create time-series sequence and tested on multiple learning rates,hidden layers, and training iterations to evaluate model perfomance.

## Dataset

The dataset contains EUR-USD OHLC values of 1-day interval.

- Datset fetched from https://raw.githubusercontent.com/preerao7/FOREX-Rate-Prediction-using-RNN/refs/heads/main/dataset/EURUSD_D1_Sorted.csv'

## Installation & Setup

### Clone the Repository

git clone https://github.com/preerao7/FOREX-Rate-Prediction-using-RNN.git

cd FOREX-Rate-Prediction-using-RNN

pip install -r requirements.txt

python main.py 

or Open and run in Jupyter notebook or google collab 

RRNForex.ipynb 


### model

Model uses LSTM and GRU in processing layers

Data is preprocessed and scaled using min-max scaling before training

The following hyperparameters are tested:
  - Learning Rates: `[0.01, 0.1]`
  - Hidden Layers: `[1, 2, 3]`
  - Training Iterations: `[50, 100, 150, 200]`
  - Time Steps: `[2, 4]`

**Final Performance Metrics:**
| Learning Rate | Hidden Layers | Time Step | R² Score | RMSE |
|--------------|--------------|----------|---------|------|
| 0.01 | 1 | 2 | 0.82 | 1.23 |
| 0.1 | 2 | 30 min | 0.89 | 0.98 |







