from ucimlrepo import fetch_ucirepo 
import json
import pandas as pd
  
'''Fetch Dataset'''

statlog_german_credit_data = fetch_ucirepo(id=144)  
X = statlog_german_credit_data.data.features.copy()
y = statlog_german_credit_data.data.targets.copy()
X.to_csv('data/features.csv', index=False)
y.to_csv('data/labels.csv', index=False)

'''Feature Engineering'''

# Load a JSON schema containing metadata
with open("data_metdata.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

# Get feature groups from schema
categorical_features = [
    col for col, meta in schema.items()
    if meta["role"] == "Feature" and meta["type"] == "Categorical"
]

binary_features = [
    col for col, meta in schema.items()
    if meta["role"] == "Feature" and meta["type"] == "Binary"
]

numerical_features = [
    col for col, meta in schema.items()
    if meta["role"] == "Feature" and meta["type"] == "Integer"
]

# Sensitive attributes for fairness analysis
sensitive_features = [
    col for col, meta in schema.items()
    if meta["role"] == "Feature" and meta["demographic"] is not None
]

print("Categorical features:", categorical_features)
print("Binary features:", binary_features)
print("Numerical features:", numerical_features)
print("Sensitive features:", sensitive_features)