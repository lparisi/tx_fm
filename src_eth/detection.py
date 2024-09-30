import os
os.environ["CUDA_VISIBLE_DEVICES"]="7" ## Specify the GPU to use

import sys
sys.path.append('/home/xww0766/tx_fm/')

import json
import logging
import math
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

import torch
from torch.nn.functional import softmax
import torch.nn.functional as F

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import AutoTokenizer, LongformerForMaskedLM
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)

from src_eth.foundation_model import replace_roberta_attn

np.random.seed(42)


# Load Model and tokenizer
tokenizer_kwargs = {
        "cache_dir": None,
        "use_fast": True,
        "revision": 'main',
        "use_auth_token": False,
}

token_path = '../config/tokenizer' # this can be fixed
tokenizer = AutoTokenizer.from_pretrained(token_path, **tokenizer_kwargs)

config_kwargs = {
        "cache_dir": None,
        "revision": 'main',
        "use_auth_token": False
}

replace_roberta_attn(False, True)
model_path = '../model/rope_roberta' # this can be fixed

config = AutoConfig.from_pretrained(model_path, **config_kwargs)
fm_model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            from_tf=bool(".ckpt" in model_path),
            config=config,
            cache_dir=None,
            revision='main',
            use_auth_token=False,
            low_cpu_mem_usage=False,
)


def compute_abnormal_score(data_dir, detection_threshold=75, model=fm_model, tokenizer=tokenizer,
                           random_masking=True, g=15, s=3, masked_tokens=[], seq_len=1024):

    """
    :param data_dir: path to the pre-tokenization data of testing TXs
    :param detection_threshold: thredshold for abnormal tx; current value is 62, selected based on the existing testing data
    :param model: model to use
    :param random_masking: whether to conduct random masking
    :param g: if random masking, mask percentage 
    :param s: select top s predicted tokens
    :param masked_tokens: if not random masking, the token id of the tokens to mask
    """

    # Open the pre-tokenization data 
    with open(data_dir+'/preprocessed_tx.txt', 'r') as file:
        lines = file.readlines()

    # read tx path
    with open(data_dir+'/tx_path.txt', 'r') as file:
        path_lines = file.readlines()

    model = model.cuda()
    model = model.eval()

    scores = {} # abnormal scores of all tx
    abnormal = {} # the dected abnormal tx

    for data, name in tqdm(zip(lines, path_lines)):
        # tokenization 
        input_ids = tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)["input_ids"]
        input_ids = input_ids[0]

        # padding
        current_size = input_ids.size(0)
        remainder = current_size % 8
        padding_size = (8 - remainder) % 8
        input_ids = F.pad(input_ids, (0, padding_size), 'constant', 1)
        # padding_size = 0

        masked_tokens = []
        if random_masking:
            # Randomly mask g% of tokens
            masked_indices = np.random.choice(np.arange(1, len(input_ids) - padding_size), size=int(len(input_ids) * g / 100), replace=False)
            for idx in masked_indices:
                masked_tokens.append((idx, input_ids[idx].item()))
                input_ids[idx] = tokenizer.mask_token_id
        else:
            for token_id in masked_tokens:
                for idx in len(input_ids):
                    if input_ids[idx].item() == token_id:
                        masked_tokens.append((idx, input_ids[idx].item()))
                        input_ids[idx] = tokenizer.mask_token_id

        attention_mask = torch.ones((len(input_ids)))
        if padding_size != 0:
            attention_mask[-padding_size:] = 0
        attention_mask = attention_mask.cuda().unsqueeze(0)

        # Predict masked tokens
        input_tensor = torch.tensor([input_ids.tolist()])
        input_tensor = input_tensor.cuda()
        logits = model(input_ids=input_tensor, attention_mask=attention_mask).logits

        # Count abnormal tokens
        abnormal_token_count = 0
        for idx, original_token_id in masked_tokens:
            probs = softmax(logits[0, idx], dim=0)
            values, predictions = probs.topk(s) # select top s predicted tokens

            if original_token_id not in predictions:
                abnormal_token_count += 1

        scores[name] = abnormal_token_count

        if abnormal_token_count >= detection_threshold:
            abnormal[name] = abnormal_token_count
    
    scores = sorted(scores.items(), key=lambda x: x[1])
    abnormal = sorted(abnormal.items(), key=lambda x: x[1])
    
    return scores, abnormal


if __name__ == '__main__':

    data_dir = '../preprocessed_data/exp_test'
    scores_all_tx, abnormal = compute_abnormal_score(data_dir)

    # data_dir = '../preprocessed_data/malicious'
    # scores_all_tx, abnormal = compute_abnormal_score(data_dir)
    #
    # data_dir = '../preprocessed_data/benign_testing'
    # scores_all_tx_b, abnormal_b = compute_abnormal_score(data_dir)
    #
    # data_dir = '../preprocessed_data/benign_training'
    # scores_all_tx_b_t, abnormal_b_t = compute_abnormal_score(data_dir)
    print('done')









"""
Archived code for computing metrics 
thresholds = [2, 4, 6]

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
"""