import numpy as np
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix

def tp_fp_fn_tn_no_response(cm):
    true_positives = np.diag(cm)
    false_positives = np.sum(cm, axis=0) - true_positives
    false_negatives = np.sum(cm, axis=1) - true_positives
    true_negatives = np.sum(cm) - (true_positives + false_positives + false_negatives)
    return true_positives, false_positives, false_negatives, true_negatives

def sensitivity_ci(cm, dims=np.array([1]), alpha=0.05):
    counts = tp_fp_fn_tn_no_response(cm)
    TP, FP, FN, TN = (c[dims].sum() for c in counts)

    if (TP + FN) == 0:
        return 0, (0, 0)

    sensitivity = TP / (TP + FN)

    sensitivity_CI = proportion_confint(TP, TP + FN, method='wilson', alpha=alpha)

    return sensitivity, sensitivity_CI

def specificity_ci(cm, dims=np.array([1]), alpha=0.05):
    counts = tp_fp_fn_tn_no_response(cm)
    TP, FP, FN, TN = (c[dims].sum() for c in counts)

    if (TN + FP) == 0:
        return 0, (0, 0)

    specificity = TN / (TN + FP)
    specificity_CI = proportion_confint(TN, TN + FP, method='wilson', alpha=alpha)

    return specificity, specificity_CI

def precision_ci(cm, dims=np.array([1]), alpha=0.05):
    counts = tp_fp_fn_tn_no_response(cm)
    TP, FP, FN, TN = (c[dims].sum() for c in counts)
    
    if (TP + FP) == 0:
        return 0, (0, 0)
    
    precision = TP / (TP + FP)
    precision_CI = proportion_confint(TP, TP + FP, method='wilson', alpha=alpha)
    
    return precision, precision_CI

def recall_ci(cm, dims=np.array([1]), alpha=0.05):
    counts = tp_fp_fn_tn_no_response(cm)
    TP, FP, FN, TN = (c[dims].sum() for c in counts)
    
    if (TP + FN) == 0:
        return 0, (0, 0)

    recall = TP / (TP + FN)
    recall_CI = proportion_confint(TP, TP + FN, method='wilson', alpha=alpha)
    
    return recall, recall_CI

def mcnemars_test(predsA, predsB, gts, label=None):
    if label is not None:
        gt_mask = gts == label
        gts = gts[gt_mask]
        predsA = predsA[gt_mask]
        predsB = predsB[gt_mask]

    predsA, predsB, gts = np.array(predsA), np.array(predsB), np.array(gts)
    
    # Calculate n00, n01, n10, n11
    n11 = np.sum((predsA == gts) & (predsB == gts))
    n01 = np.sum((predsA != gts) & (predsB == gts))
    n10 = np.sum((predsA == gts) & (predsB != gts))
    n00 = np.sum((predsA != gts) & (predsB != gts))

    # Create the McNemar table
    mcnemar_table = np.array([[n11, n01],
                            [n10, n00]])

    # Perform McNemar's test
    result = mcnemar(mcnemar_table, exact=False)

    # print(f"McNemar's test p-value: {result.pvalue}")
    # print(f"McNemar's table:\n{mcnemar_table}")
    return (result.statistic, result.pvalue)

def sens_spec_no_response(cm):
    # True Positives (TP): Directly obtained from the diagonal
    cm_classes = cm[:-1,:-1]
    true_positives = np.diag(cm_classes)
    false_negatives = np.sum(cm[:-1,:], axis=1) - true_positives
    
    # True Negatives (TN): If the class wasn't positive and it successfully confirmed it was another class
    true_negatives = cm_classes.sum() - cm_classes.sum(0) - cm_classes.sum(1) + true_positives
    all_negatives = cm[:-1].sum() - cm[:-1].sum(1)
    
    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / all_negatives

    return sensitivity, specificity

def precision_no_response(cm):
    cm_classes = cm[:-1,:-1]
    true_positives = np.diag(cm_classes)
    false_positives = cm_classes.sum(0) - true_positives
    
    mask = true_positives + false_positives > 0
    
    precision = np.zeros(mask.shape)
    precision[mask] = true_positives[mask] / (true_positives[mask] + false_positives[mask])

    return precision

def recall_no_response(cm):
    cm_classes = cm[:-1,:-1]
    true_positives = np.diag(cm_classes)
    false_negatives = np.sum(cm[:-1,:], axis=1) - true_positives
    
    mask = true_positives + false_negatives == 0
    recall = np.zeros(mask.shape)
    
    recall[mask] = true_positives[mask] / (true_positives[mask] + false_negatives[mask])

    return recall

def per_class_accuracy_no_response(cm):
    accuracy = np.diag(cm[:-1]) / cm[:-1].sum(1)

    return accuracy

def overall_accuracy_no_response(cm):
    accuracy = np.diag(cm[:-1]).sum() / cm[:-1].sum()

    return np.array([accuracy])

def f1_no_response(cm, dims=np.array([1])):
    precision = precision_ci(cm, dims=dims)[0]
    recall = recall_ci(cm, dims=dims)[0]

    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 / ((1 / precision) + (1 / recall))

    return f1

def cm_no_response(y_true, y_pred, options):
    return confusion_matrix(y_true, y_pred, labels=options + ['Invalid response'])

def bootstrap_f1_confidence_interval(y_true, y_pred, options, positive_classes=['present'], n_bootstraps=1000, ci=95):
    bootstrapped_scores = []

    positive_ix = [i for i, element in enumerate(options) if element in positive_classes]

    overall_cm = cm_no_response(y_true, y_pred, options)
    overall_score = f1_no_response(overall_cm, dims=positive_ix)

    if not n_bootstraps:
        return overall_score
    
    for _ in range(n_bootstraps):
        # Randomly sample with replacement from the original data
        indices = resample(np.arange(len(y_true)))
        if len(np.unique(np.array(y_true)[indices])) < 2:
            # Skip this iteration if the sample doesn't contain both classes
            continue
        cm = cm_no_response(np.array(y_true)[indices].tolist(), np.array(y_pred)[indices].tolist(), options)
        score = f1_no_response(cm, dims=positive_ix)
        bootstrapped_scores.append(score)
    
    alpha = 100 - ci
    lower = np.percentile(bootstrapped_scores, alpha / 2.)
    upper = np.percentile(bootstrapped_scores, 100 - alpha / 2.)

    return overall_score, (lower, upper)
