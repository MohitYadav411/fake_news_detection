import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Computes standard classification metrics.
    Returns a dictionary with accuracy, precision, recall, and f1.
    We assume binary classification where 1=Fake, 0=Real.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred))
    }

def plot_confusion_matrix(y_true, y_pred, labels=("Fake", "Real")) -> plt.Figure:
    """
    Generates and returns a confusion matrix figure.
    Because our internal labeling is 1=Fake, 0=Real, we map them accordingly.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Generate confusion matrix array. Assuming 1 (Fake) and 0 (Real).
    # We will pass labels=[1, 0] so it matches our ("Fake", "Real") display labels
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    
    return fig
