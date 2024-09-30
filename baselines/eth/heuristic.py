import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# get the length of the txs 
test_file = '/home/xww0766/ood_transaction/baselines/data/test_tst.txt'
with open(file_path, 'r') as file:
    txs = file.readlines()

scores = [len(tst.strip()) for tst in txs]

gt_labels = ['benign'] * 709 + ['malicious'] * 10

thresholds = [5, 10, 15]
# Initialize lists to store metrics for each threshold
fp_list, fpr_list = np.zeros(len(thresholds)), np.zeros(len(thresholds))
precision_list, recall_list = np.zeros(len(thresholds)), np.zeros(len(thresholds))

top_fp_list, top_fpr_list = np.zeros(3), np.zeros(3)
top_precision_list, top_recall_list = np.zeros(3), np.zeros(3)

success_list = np.zeros(len(thresholds))
top_success_list = np.zeros(3)

scores_np = np.array(scores)
sorted_indices = np.argsort(scores_np)[::-1]

for threshold_index, threshold_value in enumerate(thresholds):

    predicted_labels = ['benign' for score in scores]
    _len = min(threshold_value, len(scores))
    for j in range(_len):
        predicted_labels[sorted_indices[j]] = 'malicious'
        
    tp = sum((label == 'malicious') and (pred_label == 'malicious') for label, pred_label in zip(gt_labels, predicted_labels))
    fp = sum((label == 'benign') and (pred_label == 'malicious') for label, pred_label in zip(gt_labels, predicted_labels))
    fn = sum((label == 'malicious') and (pred_label == 'benign') for label, pred_label in zip(gt_labels, predicted_labels))
    fpr = fp / (sum(label == 'benign' for label in gt_labels))
    # Precision, Recall
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp * 1.0 / (tp + fp)
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp * 1.0 / (tp + fn)
    
    print(fp, sum(label == 'benign' for label in gt_labels), fpr)
    print(tp, tp + fp, precision)
    print(tp, tp + fn, recall)

    fp_list[threshold_index] += fp
    fpr_list[threshold_index] += fpr
    precision_list[threshold_index] += precision
    recall_list[threshold_index] += recall

# Calculate the averages
for threshold_index, threshold_value in enumerate(thresholds):
    print("~~~~~~~~~~~~~~~~threshold_value={}~~~~~~~~~~~~~~~~~~~~~".format(threshold_value))
    avg_fp = fp_list[threshold_index] / 1.0
    avg_fpr = fpr_list[threshold_index] / 1.0
    avg_precision = precision_list[threshold_index] / 1.0
    avg_recall = recall_list[threshold_index] / 1.0

    print(f"Average false positive samples: {avg_fp}")
    print(f"Average false positive rate: {avg_fpr:.4f}")
    print(f"Average recall rate: {avg_recall:.4f}")
    print(f"Average precision rate: {avg_precision:.4f}")
