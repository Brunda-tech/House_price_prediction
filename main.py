import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
data = pd.read_csv("Housing.csv")

# Convert text columns to numbers
le = LabelEncoder()

data["Location"] = le.fit_transform(data["Location"])
data["Condition"] = le.fit_transform(data["Condition"])
data["Garage"] = le.fit_transform(data["Garage"])

# Features and Target
X = data.drop("Price", axis=1)
y = data["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model =RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Predict a new house
new_house = [[1001, 2500, 3, 2, 2, 2015, 0, 2, 1]]
prediction = model.predict(new_house)

print("Predicted Price:", prediction[0])

import joblib
joblib.dump(model,"house_price_model.pkl")
print("Model Saved Successfully!")