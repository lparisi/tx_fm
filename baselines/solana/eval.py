# tokenize_task.py
import sys
import json
import logging
import math
import os
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
from datasets import load_dataset
from tqdm import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    is_torch_tpu_available,
    set_seed,
)

import pandas as pd
import torch
from torch.nn.functional import softmax
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

np.random.seed(42)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import AutoTokenizer, LongformerForMaskedLM

# Load Model and tokenizer
tokenizer_kwargs = {
    "cache_dir": None,
    "use_fast": True,
    "revision": 'main',
    "use_auth_token": False,
}

token_path = './data/tokenizer'
tokenizer = AutoTokenizer.from_pretrained(token_path, **tokenizer_kwargs)

def write_file(output_file_path, data):
    with open(output_file_path, 'w') as output_file:
        for value in data:
            output_file.write(f"{value}\n")

config_kwargs = {
    "cache_dir": None,
    "revision": 'main',
    "use_auth_token": False
}

model_dirs = ['/home/xww0766/ood_transaction/baselines/output/checkpoint-11000']
# Open the second file in read mode
with open('/home/xww0766/ood_transaction/baselines/data/test_tst.txt', 'r') as file:
    # Read all lines from the second file
    lines = file.readlines()
gt_labels = ['benign'] * 709 + ['malicious'] * 10

batch_size = 10
max_seq_len = 1024
all_recall_rate, all_precision_rate = [[],[],[]], [[],[],[]]

for model_id, model_path in enumerate(tqdm(model_dirs)):
    config = AutoConfig.from_pretrained(model_path, **config_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_path),
        config=config,
        cache_dir=None,
        revision='main',
        use_auth_token=False,
        low_cpu_mem_usage=False,
    )
    model = model.cuda()
    model = model.eval()
    scores, log_probs = [], []

    for i in tqdm(range(0, len(lines), batch_size), desc="Batch Processing"):
        batch = lines[i:i+batch_size]

        # Tokenize the batch
        batch_encoding = tokenizer(batch, return_tensors="pt", padding='max_length', truncation=True, max_length=max_seq_len)
        input_ids = batch_encoding['input_ids']
        attention_masks = batch_encoding['attention_mask']
        labels = input_ids.clone()

        input_ids, attention_masks = input_ids.cuda(), attention_masks.cuda()
        logits = model(input_ids, attention_mask=attention_masks).logits

        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous().cuda()

        # Mask out the padding in labels
        shifted_attention_mask = attention_masks[..., 1:].contiguous()

        log_probs_batch = F.log_softmax(shifted_logits, dim=-1)
        log_likelihoods = log_probs_batch.gather(2, shifted_labels.unsqueeze(2)).squeeze(2)
        log_likelihoods = log_likelihoods * shifted_attention_mask

        log_likelihood_per_sentence = log_likelihoods.sum(dim=1).tolist()
        log_probs.extend(log_likelihood_per_sentence)

    thresholds = [5, 10, 15]
    # Initialize lists to store metrics for each threshold
    fp_list, fpr_list = np.zeros(len(thresholds)), np.zeros(len(thresholds))
    precision_list, recall_list = np.zeros(len(thresholds)), np.zeros(len(thresholds))

    top_fp_list, top_fpr_list = np.zeros(3), np.zeros(3)
    top_precision_list, top_recall_list = np.zeros(3), np.zeros(3)

    success_list = np.zeros(len(thresholds))
    top_success_list = np.zeros(3)

    log_probs_np = np.array(log_probs) * -1.0
    sorted_indices = np.argsort(log_probs_np)[::-1]

    for threshold_index, threshold_value in enumerate(thresholds):

        predicted_labels = ['benign' for score in log_probs]
        _len = min(threshold_value, len(log_probs))
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
        
        all_recall_rate[threshold_index].append(avg_recall)
        all_precision_rate[threshold_index].append(avg_precision)
        
        print(f"Average false positive samples: {avg_fp}")
        print(f"Average false positive rate: {avg_fpr:.4f}")
        print(f"Average recall rate: {avg_recall:.4f}")
        print(f"Average precision rate: {avg_precision:.4f}")
