import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import pickle

def train_models():
    # Load and prepare the data
    data = pd.read_csv('data/transactions.csv')
    
    # Train the category model
    category_model = RandomForestClassifier()
    category_model.fit(data['description'].values.reshape(-1, 1), data['category'])
    
    # Train the expense prediction model
    X = pd.to_datetime(data['date']).astype(int).values.reshape(-1, 1)
    expense_model = LinearRegression()
    expense_model.fit(X, data['amount'])
    
    # Save the models
    with open('category_model.pkl', 'wb') as f:
        pickle.dump(category_model, f)
    with open('expense_model.pkl', 'wb') as f:
        pickle.dump(expense_model, f)

def load_models():
    with open('category_model.pkl', 'rb') as f:
        category_model = pickle.load(f)
    with open('expense_model.pkl', 'rb') as f:
        expense_model = pickle.load(f)
    return category_model, expense_model
