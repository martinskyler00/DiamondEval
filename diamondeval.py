import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

training_data = pd.read_csv('training.csv')
offers_data = pd.read_csv('offers.csv')

relevant_columns = ['Carats', 'Clarity', 'Color', 'Regions', 'Price']
training_data_filtered = training_data[relevant_columns].dropna()
x = training_data_filtered.drop('Price', axis = 1)
y = training_data_filtered['Price']

categorical_features = ['Clarity', 'Color', 'Regions']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder = 'passthrough'
)

model = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators = 100, random_state = 42))
])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
model.fit(x_train, y_train)

estimated_offers = model.predict(offers_data)
offers_data['Estimated_Offers'] = estimated_offers

mean_offer = offers_data['Estimated_Offers'].mean()
std_dev_offer = offers_data['Estimated_Offers'].std()

lower_bound = mean_offer - std_dev_offer
upper_bound = mean_offer + std_dev_offer

conflict_regions = ['Angola', 'DR Congo', 'Zimbabwe']
offers_data['Adjusted_Offers'] = offers_data.apply(
    lambda row: 0 if row['Regions'] in conflict_regions or not (lower_bound <= row['Estimated_Offers'] <= upper_bound) else row['Estimated_Offers'],
    axis = 1
)

merged_data = offers_data.merge(training_data[['id', 'Retail']], on = 'id', how = 'left')
valid_offers = merged_data[merged_data["Adjusted Offers"] > 0]
valid_offers['Potential_Income'] = valid_offers['Retail'] - vaild_offers['Adjusted_Offers']
sorted_diamonds = valid_offers.sort_values(by = 'Potential_Income', ascending = False)

budget = 5_000_000
selected_diamonds = []
cumulative_cost = 0

for _, row in sorted_diamonds.iterrows():
    if cumulative_cost + row['Adjusted_Offers'] <= budget:
        selected_diamonds.append(row['id'])
        cumulative_cost += row['Adjusted_Offers']
    else:
        break
    
offers_data['Offers'] = offers_data.apply(
    lambda row: row['Adjusted_Offers'] if row['id'] in selected_diamonds else 0, axis = 1
)

offers_data.to_csv('final_offers.csv', index = False)

