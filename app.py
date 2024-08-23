from flask import Flask, render_template, request, jsonify
import pandas as pd
import models

app = Flask(__name__)
category_model, expense_model = models.load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard', methods=['POST'])
def dashboard():
    transactions = pd.read_csv('data/transactions.csv')
    categories = category_model.predict(transactions['description'].values.reshape(-1, 1))
    predictions = expense_model.predict(pd.to_datetime(transactions['date']).astype(int).values.reshape(-1, 1))
    transactions['predicted_category'] = categories
    transactions['predicted_amount'] = predictions

    total_expense = transactions['amount'].sum()
    future_expense = predictions.sum()
    
    return render_template('dashboard.html', tables=[transactions.to_html(classes='table table-striped', index=False)],
                           total_expense=total_expense, future_expense=future_expense)

if __name__ == '__main__':
    app.run(debug=True)
