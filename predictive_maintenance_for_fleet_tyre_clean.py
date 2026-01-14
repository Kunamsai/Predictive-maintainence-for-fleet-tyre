import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error


# Load dataset (Excel). If not available, create a small sample dataset for demo.
FILE_PATH = r"C:\Users\kunam\Downloads\Tyre Monitoring Data.xlsx"

if os.path.exists(FILE_PATH):
    try:
        data = pd.read_excel(FILE_PATH)
    except Exception:
        data = pd.read_csv(FILE_PATH)
else:
    # Fallback sample data
    rng = np.random.default_rng(42)
    n = 12
    today = pd.Timestamp(datetime.now())
    data = pd.DataFrame({
        'Tyre ID': [f'T{idx+1:02d}' for idx in range(n)],
        'Installation Date': [(today - pd.to_timedelta(rng.integers(30, 3650), unit='D')).date() for _ in range(n)],
        'PSI': rng.normal(100, 8, size=n).round(1),
        'Tyre Depth (mm)': rng.uniform(2.5, 8.0, size=n).round(1),
        'Temperature (°C)': rng.uniform(15, 40, size=n).round(1),
    })

# Define average usage per day (km/day)
avg_km_per_day = 100

# Preprocessing: calculate days in use and total kilometers travelled
data['Installation Date'] = pd.to_datetime(data['Installation Date'])
data['Days In Use'] = (pd.Timestamp(datetime.now()) - data['Installation Date']).dt.days
data['Total Kilometers'] = data['Days In Use'] * avg_km_per_day


# Calculate condition score to use as target for classification model
def calculate_condition_score(psi, depth, temp):
    psi_factor = 1 if 90 <= psi <= 110 else 0
    if depth > 6:
        depth_factor = 1
    elif 3 <= depth <= 6:
        depth_factor = 0.5
    else:
        depth_factor = 0
    temp_factor = 1 if 20 <= temp <= 35 else 0
    return psi_factor + depth_factor + temp_factor


# Apply condition score and assign condition labels
data['ConditionScore'] = data.apply(lambda row: calculate_condition_score(row['PSI'], row['Tyre Depth (mm)'], row['Temperature (°C)']), axis=1)
data['Condition'] = data['ConditionScore'].apply(lambda score: 'Good' if score == 3 else 'Average' if score >= 2 else 'Bad')


# Add 'Remaining Kilometers' based on expected lifetime
expected_lifetime = data['Condition'].map({'Good': 100000, 'Average': 75000, 'Bad': 50000})
data['Remaining Kilometers'] = expected_lifetime - data['Total Kilometers']


# Define features and targets for model training
feature_cols = ['PSI', 'Tyre Depth (mm)', 'Temperature (°C)', 'Total Kilometers']
X = data[feature_cols]
y_condition = data['Condition']
y_remaining_km = data['Remaining Kilometers']


# Split data into training and test sets (adjust test size for small datasets)
test_size = 0.2 if len(data) >= 10 else 0.4
X_train, X_test, y_condition_train, y_condition_test = train_test_split(X, y_condition, test_size=test_size, random_state=42)
_, _, y_remaining_km_train, y_remaining_km_test = train_test_split(X, y_remaining_km, test_size=test_size, random_state=42)


# Train a classifier for tire condition
condition_model = RandomForestClassifier(random_state=42)
condition_model.fit(X_train, y_condition_train)

# Train a regressor for remaining kilometers
remaining_km_model = RandomForestRegressor(random_state=42)
remaining_km_model.fit(X_train, y_remaining_km_train)


# Model evaluation
condition_preds = condition_model.predict(X_test)
remaining_km_preds = remaining_km_model.predict(X_test)
print(f"Condition Model Accuracy: {accuracy_score(y_condition_test, condition_preds):.3f}")
print(f"Remaining KM Model RMSE: {np.sqrt(mean_squared_error(y_remaining_km_test, remaining_km_preds)):.1f}")


# Evaluate each tire and generate the formatted output
results = []
for index, row in data.iterrows():
    input_data = pd.DataFrame({
        'PSI': [row['PSI']],
        'Tyre Depth (mm)': [row['Tyre Depth (mm)']],
        'Temperature (°C)': [row['Temperature (°C)']],
        'Total Kilometers': [row['Total Kilometers']]
    })

    condition_pred = condition_model.predict(input_data)[0]
    remaining_km_pred = remaining_km_model.predict(input_data)[0]

    psi_formatted = f"{row['PSI']:.1f}"
    depth_formatted = f"{row['Tyre Depth (mm)']:.1f}"
    temp_formatted = f"{row['Temperature (°C)']:.1f}"

    if condition_pred == 'Bad' or remaining_km_pred <= 0:
        results.append(f"{row['Tyre ID']} is in {condition_pred} condition with [{psi_formatted} PSI, {depth_formatted} mm, {temp_formatted}°C], so it needs to be replaced immediately.")
    else:
        results.append(f"{row['Tyre ID']} is in {condition_pred} condition with [{psi_formatted} PSI, {depth_formatted} mm, {temp_formatted}°C], so no need to replace and can travel {remaining_km_pred:.0f} km more.")


# Print the results
for result in results:
    print(result)
