from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import models
from datetime import datetime

app = Flask(__name__)

# Load the pre-trained models
category_model, expense_model = models.load_models()

# Path to the transactions CSV file
DATA_PATH = 'data/transactions.csv'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        # Get the new transaction data from the form
        date = request.form['date']
        description = request.form['description']
        amount = float(request.form['amount'])
        
        # Append the new transaction to the CSV file
        new_transaction = pd.DataFrame([[date, description, amount]], columns=['date', 'description', 'amount'])
        new_transaction.to_csv(DATA_PATH, mode='a', header=False, index=False)
        
        return redirect(url_for('dashboard'))
    
    # Load and process the transactions
    transactions = pd.read_csv(DATA_PATH)
    categories = category_model.predict(transactions['description'].values.reshape(-1, 1))
    predictions = expense_model.predict(pd.to_datetime(transactions['date']).astype(int).values.reshape(-1, 1))
    transactions['predicted_category'] = categories
    transactions['predicted_amount'] = predictions

    # Calculate total and future expenses
    total_expense = transactions['amount'].sum()
    future_expense = predictions.sum()
    
    return render_template('dashboard.html', tables=[transactions.to_html(classes='table table-striped', index=False)],
                           total_expense=total_expense, future_expense=future_expense)

if __name__ == '__main__':
    app.run(debug=True)
