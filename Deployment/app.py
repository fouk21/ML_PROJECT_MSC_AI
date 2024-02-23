from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from evaluation import evaluate_predictions
import io
import base64

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.sav")

# Define a route to render the UI
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for prediction and evaluation
@app.route('/evaluate', methods=['POST'])
def evaluate():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No file selected', 400
    
    if file:
        # Read the uploaded file
        x_test = pd.read_csv(file)
        y_test = pd.read_csv("y_test.csv")
        
        # Perform prediction
        y_pred = model.predict(x_test)
        
        # Evaluate predictions
        report = evaluate_predictions(y_test, y_pred)

        # Convert classification report to HTML
        report_html = report.replace('\n', '<br>')

        # Return the evaluation metrics
        return render_template('evaluation.html', report_html=report_html)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000)
