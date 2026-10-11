import numpy as np
import evaluate
from src.config import ID2LABEL

# Load the seqeval metric via Hugging Face evaluate
seqeval_metric = evaluate.load("seqeval")

def compute_metrics(p):
    """
    Computes strict entity-level Precision, Recall, F1, and Accuracy
    for token classification while filtering out masked labels (-100).
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (-100) and convert predictions/labels back to string tags
    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }