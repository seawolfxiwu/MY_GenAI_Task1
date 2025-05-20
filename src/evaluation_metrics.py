from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from typing import Dict, List

def calculate_metrics(y_true: List, y_pred: List, label_map: Dict) -> Dict:
    """Calculate classification metrics and confusion matrix"""
    # Map labels
    y_true_mapped = [label_map.get(ele, ele) for ele in y_true]
    y_pred_mapped = [label_map.get(ele, ele) for ele in y_pred]

    labels = list(label_map.values())

    # Confusion matrix
    cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f'Actual {l}' for l in labels],
        columns=[f'Predicted {l}' for l in labels]
    )

    # Classification report
    report = classification_report(y_true_mapped, y_pred_mapped, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    return {"Classification Report": report_df, "Confusion Matrix": cm_df}

def sentiment_eval_metrics(std_df: pd.DataFrame, prd_df: pd.DataFrame) -> Dict:
    """Calculate sentiment evaluation metrics"""
    y_true = std_df['Sentiment']
    y_pred = prd_df['Sentiment']

    label_map = {"Negative": "NEG", "Neutral": "NEU", "Positive": "POS"}
    return calculate_metrics(y_true, y_pred, label_map)

def outcome_eval_metrics(std_df: pd.DataFrame, prd_df: pd.DataFrame) -> Dict:
    """Calculate outcome evaluation metrics"""
    y_true = std_df['Outcome']
    y_pred = prd_df['Outcome']

    label_map = {"Issue Resolved": "IRD", "Not applicable.": "N/A", "Follow-up Action Needed": "FAN"}
    return calculate_metrics(y_true, y_pred, label_map)
