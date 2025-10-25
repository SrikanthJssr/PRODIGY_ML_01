# ==============================
# PRODIGY_ML_01 - Linear Regression Model for House Price Prediction
# ==============================

# ---- Importing Required Libraries ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---- Step 1: Load Dataset ----
train_data = pd.read_csv("train.csv")

print("✅ Data loaded successfully!")
print("Shape of dataset:", train_data.shape)
print(train_data.head())

# ---- Step 2: Select Important Features ----
# We will use 'GrLivArea' (living area), 'BedroomAbvGr' (bedrooms), 'FullBath' (bathrooms)
# as predictors and 'SalePrice' as the target
features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
target = "SalePrice"

# Drop rows with missing values for these columns
data = train_data[features + [target]].dropna()

# ---- Step 3: Split Data into Train & Test ----
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Step 4: Train Linear Regression Model ----
model = LinearRegression()
model.fit(X_train, y_train)

print("\n✅ Model trained successfully!")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# ---- Step 5: Make Predictions ----
y_pred = model.predict(X_test)

# ---- Step 6: Evaluate Model ----
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# ---- Step 7: Visualization ----

# Scatter plot of predicted vs actual prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

print("\n✅ Visualization complete! Task 01 successful 🚀")
