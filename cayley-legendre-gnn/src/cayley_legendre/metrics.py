from __future__ import annotations

import numpy as np


def average_precision_score_torch(y_true: np.ndarray, y_score: np.ndarray) -> float:

    from sklearn.metrics import average_precision_score
    ap_list = []
    for i in range(y_true.shape[1]):
        col_true = y_true[:, i]

        if np.all(col_true == col_true[0]):
            continue
        ap_list.append(average_precision_score(col_true, y_score[:, i]))
    return float(np.mean(ap_list)) if ap_list else float("nan")
