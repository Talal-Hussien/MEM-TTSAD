"""POT (peaks-over-threshold) evaluation with the point-adjust protocol."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from memttsad.evaluation.spot import SPOT


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    try:
        roc_auc = roc_auc_score(actual, predict)
    except:
        roc_auc = 0
    return f1, precision, recall, TP, TN, FP, FN, roc_auc


# the below function is taken from OmniAnomaly code base directly
def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def pot_eval(init_score, score, label, lm, q=1e-5, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            it should be the anomaly score of test set.
        label:
        lm (tuple): (initial level, threshold multiplier) for the dataset/model
            protocol (config.get_lm).
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t
    Returns:
        dict: pot result dict
    """
    lms = lm[0]
    retries = 0
    s = None
    while True:
        try:
            s = SPOT(q)  # SPOT object
            s.fit(init_score, score)  # data import
            s.initialize(level=lms, min_extrema=False, verbose=False)  # initialization step
        except:
            lms = lms * 0.999
            retries += 1
            if retries % 1000 == 0:
                print(f'SPOT init retry {retries}, level={lms:.6f}', flush=True)
            if retries >= 5000:
                # SPOT cannot initialize on this score distribution (e.g. a
                # channel constant in training: all init scores equal, so the
                # peak set is empty at every level and the retry loop would
                # spin forever). Fall back to a deterministic max-train-error
                # threshold; only reachable where the original code never
                # terminated, so all terminating cases stay bit-identical.
                s = None
                break
        else: break
    if s is not None:
        ret = s.run(dynamic=False)  # run
        pot_th = np.mean(ret['thresholds']) * lm[1]
        if not np.isfinite(pot_th) or pot_th > np.max(score):
            # GPD-extrapolation guard (2026-09-08). Grimshaw can return a
            # degenerate heavy-tail fit (shape >> 1, sigma ~ 1e-8), whose
            # extreme quantile is 1e25..1e34; with dynamic=False that value
            # is carried until the first exceedance and dominates the mean
            # threshold, so nothing can ever be flagged. A threshold above
            # every test score is exactly the zero-raw-positive case, so
            # this branch is a no-op for every run that produced at least
            # one raw positive. Fallback: the SPOT empirical peak threshold
            # (what SPOT itself reverts to after the first exceedance, and
            # what a healthy fit's extreme quantile equals to ~4 digits),
            # scaled by the same lm factor.
            fb = s.init_threshold * lm[1]
            print(f'POT threshold {pot_th:.6g} exceeds max test score {np.max(score):.6g} '
                  f'(degenerate GPD fit); falling back to SPOT init threshold {fb:.6g}', flush=True)
            pot_th = fb
    else:
        pot_th = np.max(init_score) * lm[1]
        print(f'SPOT init failed after {retries} retries; '
              f'falling back to max-train-score threshold {pot_th:.6g}', flush=True)
    raw_pred = (np.asarray(score) > pot_th).astype(int)
    r_t = calc_point2point(raw_pred, label)
    try:
        roc_auc_raw = roc_auc_score(np.asarray(label) > 0.1, np.asarray(score))
    except:
        roc_auc_raw = 0
    try:
        pr_auc_raw = average_precision_score(np.asarray(label) > 0.1, np.asarray(score))
    except:
        pr_auc_raw = 0
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    p_t = calc_point2point(pred, label)
    return {
        'f1': p_t[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
        'ROC/AUC': p_t[7],
        'threshold': pot_th,
        'f1_unadjusted': r_t[0],
        'precision_unadjusted': r_t[1],
        'recall_unadjusted': r_t[2],
        'roc_auc_raw': roc_auc_raw,
        'pr_auc_raw': pr_auc_raw,
    }, np.array(pred)
