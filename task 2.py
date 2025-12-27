import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Tesla.csv")

# Convert Date column if exists
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

# Fill missing values
df.fillna(method="ffill", inplace=True)

# Select features and target
features = [col for col in ["Open", "High", "Low", "Volume"] if col in df.columns]

if "Close" in df.columns:
    target = "Close"
elif "Adj Close" in df.columns:
    target = "Adj Close"
else:
    raise Exception("Close price column not found")

X = df[features]
y = df[target]

# Scale features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1,1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, shuffle=False
)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Inverse scaling
y_test_actual = scaler_y.inverse_transform(y_test)
y_pred_actual = scaler_y.inverse_transform(y_pred)

# Evaluation
mse = mean_squared_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Plot results
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label="Actual Price", color="blue")
plt.plot(y_pred_actual, label="Predicted Price", color="red")
plt.title("Tesla Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

