import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the datasets
training_data = pd.read_csv('training.csv')
offers_data = pd.read_csv('offers.csv')

# Prepare the training data for the model
relevant_columns = ['Carats', 'Clarity', 'Color', 'Regions', 'Price']
training_data_filtered = training_data[relevant_columns].dropna()
X = training_data_filtered.drop('Price', axis=1)
y = training_data_filtered['Price']

# Build and train the model
categorical_features = ['Clarity', 'Color', 'Regions']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Estimate offers for the diamonds in the offers.csv
estimated_offers = model.predict(offers_data)
offers_data['Estimated_Offers'] = estimated_offers

# Apply rules for conflict diamonds and offer amount range
conflict_regions = ['Angola', 'DR Congo', 'Zimbabwe']
offers_data['Adjusted_Offers'] = offers_data.apply(
    lambda row: 0 if row['Regions'] in conflict_regions or not (5000 <= row['Estimated_Offers'] <= 15000) else row['Estimated_Offers'],
    axis=1
)

# Select diamonds within the $5 million budget
merged_data = offers_data.merge(training_data[['id', 'Retail']], on='id', how='left')
valid_offers = merged_data[merged_data['Adjusted_Offers'] > 0]
valid_offers['Potential_Income'] = valid_offers['Retail'] - valid_offers['Adjusted_Offers']
sorted_diamonds = valid_offers.sort_values(by='Potential_Income', ascending=False)

budget = 5_000_000
selected_diamonds = []
cumulative_cost = 0

for _, row in sorted_diamonds.iterrows():
    if cumulative_cost + row['Adjusted_Offers'] <= budget:
        selected_diamonds.append(row['id'])
        cumulative_cost += row['Adjusted_Offers']
    else:
        break

offers_data['Final_Offers'] = offers_data.apply(
    lambda row: row['Adjusted_Offers'] if row['id'] in selected_diamonds else 0, axis=1
)

# Save the final version of offers.csv
offers_data.to_csv('goofytest2_offers.csv', index=False)
