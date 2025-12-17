import joblib
import sys
import json
import os
import pandas as pd

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pkl')
model = joblib.load(model_path)

# Read input from stdin (JSON)
input_data = json.loads(sys.stdin.read())

# Features: income, credit_score, credit_card_usage, education_level, family_size, age, loan_amount
features = [
    input_data['income'],
    input_data['credit_score'],
    input_data['credit_card_usage'],
    input_data['education_level'],
    input_data['family_size'],
    input_data['age'],
    input_data['loan_amount']
]

# Create DataFrame with column names
df = pd.DataFrame([features], columns=['income', 'credit_score', 'credit_card_usage', 'education_level', 'family_size', 'age', 'loan_amount'])

# Predict
prediction = model.predict(df)[0]

# Output as JSON
output = {'approved': int(prediction)}
print(json.dumps(output))