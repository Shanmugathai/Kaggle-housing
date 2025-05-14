# Required Libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

try:
  df = pd.read_csv(fn) # Assuming the uploaded file is a CSV
  print(df.head()) # Display first few rows of the DataFrame
except pd.errors.ParserError:
  print("Error: The uploaded file does not appear to be a valid CSV file.")
except FileNotFoundError:
    print(f"Error: File '{fn}' not found. Please upload a valid CSV file.")
except Exception as e:
  print(f"An unexpected error occurred: {e}")
  
# 1. Load CSV File
file_path = 'kaggle_housing_data.csv'  # <-- Update this if the file is elsewhere
df = pd.read_csv(file_path)

# 2. Quick Data Preview
print("\nFirst few rows of the dataset:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

# 3. Preprocessing
df = df.dropna()

# Set target and features
target_column = 'price'
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset.")

X = df.drop(columns=[target_column])
y = df[target_column]

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Define and tune XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("\nBest Parameters Found:")
print(grid_search.best_params_)

# 5. Evaluation on Test Set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nTest Set Evaluation:")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# 6. Cross-Validation RMSE
cv_scores = cross_val_score(best_model, X_scaled, y, scoring='neg_root_mean_squared_error', cv=5)
print(f"\nCross-validated RMSE (mean): {-np.mean(cv_scores):.4f}")

# 7. Feature Importance Plot
plt.figure(figsize=(12, 8))
xgb.plot_importance(best_model, importance_type='gain', max_num_features=20)
plt.title("Top 20 Feature Importances (by Gain)")
plt.tight_layout()
plt.show()