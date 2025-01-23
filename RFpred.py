import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('combined_gst_data_2017_2024 - Copy.csv')

# Preprocess the dataset
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Encode 'Tax Type'
data = pd.get_dummies(data, columns=['Tax Type'], drop_first=True)

# Define target and features
target = 'Grand Total'
features = [col for col in data.columns if col not in ['Date', 'Grand Total']]

X = data[features]
y = data[target]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Clean X_train and X_test
for dataset in [X_train, X_test]:
    # Replace problematic entries
    dataset.replace(r'^\s*-\s*$', np.nan, regex=True, inplace=True)
    
    # Convert to numeric, forcing invalid entries to NaN
    dataset = dataset.apply(pd.to_numeric, errors='coerce')
    
    # Fill missing values with 0
    dataset.fillna(0, inplace=True)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predict on cleaned X_test
y_pred = model.predict(X_test)

# Evaluate predictions
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")


# Feature Importance
feature_importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
feature_importance.sort_values(by='Importance', ascending=False, inplace=True)

# Visualization: Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Visualization: Actual vs. Predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', alpha=0.7)
plt.plot(y_pred, label='Predicted', alpha=0.7)
plt.title('Actual vs. Predicted GST Collections')
plt.xlabel('Samples')
plt.ylabel('GST Collection (in Millions)')
plt.legend()
plt.show()

# Visualization: Residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.axvline(0, color='red', linestyle='--')
plt.show()

# Clean state columns: Convert non-numeric entries to NaN and then compute the mean
state_columns = [col for col in data.columns if col not in ['Date', 'Grand Total', 'Year Range', 'Month', 'Year', 'Tax Type_CGST', 'Tax Type_IGST', 'Tax Type_SGST']]

# Clean and convert to numeric values for each state column
for state in state_columns:
    data[state] = pd.to_numeric(data[state], errors='coerce')  # Convert non-numeric entries to NaN

# Now we can safely compute the mean values for 2024
data_2024 = pd.DataFrame({
    'Month': [1],  # January for example (extend this for other months later)
    'Year': [2025],
})

# Add state-specific columns (using the latest data from 2024)
for state in state_columns:
    data_2024[state] = [data[data['Year'] == 2024][state].mean()]

# Now handle tax type columns (Tax Type_CGST, Tax Type_IGST, Tax Type_SGST)
data_2024['Tax Type_CGST'] = [data[data['Year'] == 2024]['Tax Type_CGST'].mean()]
data_2024['Tax Type_IGST'] = [data[data['Year'] == 2024]['Tax Type_IGST'].mean()]
data_2024['Tax Type_SGST'] = [data[data['Year'] == 2024]['Tax Type_SGST'].mean()]

# Encode 'Tax Type' and handle missing data for prediction
data_2025 = pd.get_dummies(data_2024, columns=['Tax Type_CGST', 'Tax Type_IGST', 'Tax Type_SGST'], drop_first=True)

# Ensure all columns for prediction are included (add missing columns if necessary)
for col in features:
    if col not in data_2025.columns:
        data_2025[col] = 0  # Fill missing columns with 0 to match the training data structure

# Predict the Grand Total for April 2025
predicted_grand_total_2025 = model.predict(data_2025[features])
predicted_value_april_2025 = predicted_grand_total_2025[0]
print(f"Predicted Grand Total for April 2025: {predicted_value_april_2025:.2f}")

# Create a time series plot from 2017 to 2024 for the Grand Total
plt.figure(figsize=(12, 6))

# Plot the Grand Total for 2017-2024 (actual data)
sns.lineplot(x='Date', y='Grand Total', data=data[data['Year'] >= 2017], label='Actual Grand Total (2017-2024)', marker='o', color='blue')

# Add the predicted value for April 2025 (as a separate point)
# For the x-axis, April 2025 will be placed as the next data point (after the last month in 2024)
plt.scatter(pd.to_datetime('2025-04-01'), predicted_value_april_2025, color='red', label=f'Predicted April 2025 ({predicted_value_april_2025:.2f})', s=100)

# Customizing the plot
plt.title('Grand Total Month-on-Month (2017-2024) with Predicted April 2025')
plt.xlabel('Date')
plt.ylabel('Grand Total (in Millions)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()