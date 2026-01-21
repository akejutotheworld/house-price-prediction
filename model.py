import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression


# Load dataset
data = pd.read_csv('data/housing.csv')


X = data[['area', 'bedrooms', 'bathrooms', 'location_score']]
y = data['price']


# Train model
model = LinearRegression()
model.fit(X, y)


# Save model
with open('model/house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)


print("Model trained and saved successfully")