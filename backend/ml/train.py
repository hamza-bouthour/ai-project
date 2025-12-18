import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


os.makedirs("../data", exist_ok=True)
#  synthetic dataset
np.random.seed(42)
n_samples = 1000

# features
income = np.random.normal(50000, 15000, n_samples)
credit_score = np.random.normal(650, 100, n_samples)
credit_card_usage = np.random.normal(0.5, 0.2, n_samples)
education_level = np.random.choice([0, 1, 2], n_samples)  # 0: high school, 1: bachelor's, 2: master's
family_size = np.random.poisson(3, n_samples)
age = np.random.normal(35, 10, n_samples)
loan_amount = np.random.normal(20000, 10000, n_samples) # including loan_amount as feature

# Target: loan approval (1: approved, 0: not)
# Rules: multiple factors
loan_to_income = loan_amount / np.maximum(income, 1)
approved = (
    (credit_score > 620) &
    (income > 35000) &
    (loan_to_income < 0.45) &
    (credit_card_usage < 0.75) &
    (age >= 18)
).astype(int)

data = pd.DataFrame({
    'income': income,
    'credit_score': credit_score,
    'credit_card_usage': credit_card_usage,
    'education_level': education_level,
    'family_size': family_size,
    'age': age,
    'loan_amount': loan_amount,
    'approved': approved
})

data.to_csv('../data/loan_data.csv', index=False)


X = data.drop('approved', axis=1)
y = data['approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=2000))
    ]),
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier())
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC())
    ]),
    'DecisionTree': DecisionTreeClassifier()
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {acc:.4f}')
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# best model
joblib.dump(best_model, 'best_model.pkl')
print(f'Best model: {type(best_model).__name__} with accuracy {best_accuracy:.4f}')