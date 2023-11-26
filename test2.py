import pandas as pd

# Load the datasets
training_data = pd.read_csv('training.csv')
offers_data = pd.read_csv('offers.csv')

# Explore the first few rows
print(training_data.head())
print(offers_data.head())

# Basic statistics and info
print(training_data.describe())
print(training_data.info())
print(offers_data.info())



import matplotlib.pyplot as plt
import seaborn as sns


sns.scatterplot(data=training_data, x='Carats', y='Retail')

sns.pairplot(training_data)