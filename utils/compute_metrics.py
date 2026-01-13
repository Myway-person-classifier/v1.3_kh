import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def get_metric(args):
    def compute_metrics(eval_preds):
        metric = dict()
        logits = eval_preds.predictions
        labels = eval_preds.label_ids

        # [수정] NaN 방어
        if np.isnan(logits).any() or np.isinf(logits).any():
            logits = np.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)

        # Labels binarize
        labels = (labels > 0.5).astype(int)

        # Metrics
        preds = (logits > 0).astype(int)
        metric['accuracy'] = accuracy_score(labels, preds)
        metric['f1_score'] = f1_score(labels, preds, average='macro', zero_division=0)
        metric['roc_auc'] = roc_auc_score(labels, logits)
        
        return metric
    return compute_metrics