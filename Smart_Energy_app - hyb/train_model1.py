import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load cleaned dataset
df = pd.read_csv("cleaned_energy_data.csv")

# Select features
X = df[['Household_Size', 'Monthly_Income', 'Appliance_Count', 'Avg_Daily_Usage_Hours']]
y = df['Monthly_Energy_Consumption']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training complete! model.pkl created successfully.")
