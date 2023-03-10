# Dynamic Asset Allocation with LSTM
This project aims to predict dynamic asset allocation using a Long Short-Term Memory (LSTM) model. Asset allocation refers to the composition of an investment portfolio, and in this project we are looking at assets such as stocks, bonds, and cash.

# Project organization
This project consists of the following

* data: A folder where asset price data for stocks, bonds, cash, etc. is stored.
* models: A folder where code related to the LSTM model is stored.
* src: A folder where utility functions, such as data preprocessing, are stored.
* notebooks: The folder where Jupyter Notebook files are stored.
* main.py: The file that serves as the entry point for the program.

# Data
* Kospi : Log return data for the Korean stock market index
* short_bond : Log Return with Interest for short term Corporate bond AA rating
* mid_bond : Log Return with Interest for mid term Corporate bond AA rating
* long_bond : Log Return with Interest for long term Corporate bond AA rating
* iau(gold ETF) : Log return data for the gold ETF
* dbc(raw materials ETF) : Log return data for raw materials ETF
* vnq(real estate ETF) :Log return data for real estate ETF

# Model Architecture
Set the Markowitz portfolio optimization function as the loss function of the deep learning model to perform dynamic asset allocation.

![model](model.png)

# Evaluation





 
