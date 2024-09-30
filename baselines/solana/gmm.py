import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Train / eval feature
csv_file = '/home/xww0766/ood_transaction/baselines/paragraph-vectors/data/tst_model.dbow_numnoisewords.2_vecdim.100_batchsize.256_lr.0.001000_epoch.43_loss.0.593990.csv'
df = pd.read_csv(csv_file, header=0)
vectors = df.to_numpy()
X, feats = vectors[:3383,:], vectors[3383:,:]
# Step 2: Fit GMMs with varying number of clusters
n_clusters = np.arange(1, 11)
models = [GaussianMixture(n_components=n, covariance_type='full', random_state=0).fit(X) for n in n_clusters]

# Step 3: Calculate BIC for each model
bics = [m.bic(X) for m in models]

# Step 4: Select the model with the lowest BIC
best_model_index = np.argmin(bics)
best_gmm = models[best_model_index]

# Optional: Plot BIC scores
plt.plot(n_clusters, bics, marker='o')
plt.title('BIC Scores by Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC')
plt.show()

# Display the optimal number of clusters
print(f"The optimal number of clusters based on BIC is: {n_clusters[best_model_index]}")

scores = -1.0 * best_gmm.score_samples(feats)
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
