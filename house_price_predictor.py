import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Dummy data: house size (sq ft) and number of rooms vs. price (thousands)
data = {
    'size': [1500, 1600, 1700, 1800, 1900, 2000],
    'rooms': [3, 3, 4, 4, 5, 5],
    'price': [200, 220, 250, 270, 300, 320]
}
df = pd.DataFrame(data)

# Features (X) and target (y)
X = df[['size', 'rooms']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)  # R^2 score
print(f"Model R^2 Score: {accuracy:.2f}")
print(f"Predicted price for test data: {y_pred[0]:.2f}k")
print("Orion’s house price predictor—ready for client data!")