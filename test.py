import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the datasets
training_data = pd.read_csv('training.csv')
offers_data = pd.read_csv('offers.csv')

# Exploratory Data Analysis
print(training_data.describe())
print(training_data.info())
sns.pairplot(training_data)
plt.show()

# Feature Engineering
training_data = pd.get_dummies(training_data, columns=['cut', 'color', 'clarity'])
training_data = training_data.fillna(method='ffill')

# Splitting the data for model training
X = training_data.drop(columns=['retail_price', 'purchase_price'])
y = training_data['retail_price']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Model evaluation
predictions = model.predict(X_val)
print(f"RMSE: {mean_squared_error(y_val, predictions, squared=False)}")

# Preparing offers_data for prediction
offers_data_processed = pd.get_dummies(offers_data, columns=['cut', 'color', 'clarity'])
# Align columns with training data
offers_data_processed = offers_data_processed.reindex(columns = X_train.columns, fill_value=0)

# Predicting retail prices for offers
offers_data['predicted_retail_price'] = model.predict(offers_data_processed)

# Making offers based on predictions
profit_margin = 0.20  # Example margin
budget = 5000000
offers_data['offer'] = offers_data['predicted_retail_price'] * (1 - profit_margin)

# Ensuring budget constraint
total_offers = offers_data['offer'].sum()
if total_offers > budget:
    scale_factor = budget / total_offers
    offers_data['offer'] *= scale_factor

# Save the offers file with offers
offers_data.to_csv('final_offers.csv', index=False)

print("Offers have been calculated and saved to 'final_offers.csv'.")
