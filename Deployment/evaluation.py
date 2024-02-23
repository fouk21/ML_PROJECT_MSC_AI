import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_predictions(y_true, y_pred):
    
    # Generate classification report
    report = classification_report(y_true, y_pred)
    
    # Print classification report
    print("Classification Report:")
    print(report)

    return report
