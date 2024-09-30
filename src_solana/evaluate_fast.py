# tokenize_task.py
import sys
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
from datasets import load_dataset
from tqdm import tqdm

sys.path.append('/home/xww0766/tx_fm/')
from src_solana.foundation_model import replace_roberta_attn

import transformers
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
os.environ["CUDA_VISIBLE_DEVICES"]="7"
# Load Model and tokenizer
tokenizer_kwargs = {
        "cache_dir": None,
        "use_fast": True,
        "revision": 'main',
        "use_auth_token": False,
}

token_path = './tokenizer'
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
# Build model and proj
replace_roberta_attn(True, True)

model_dirs = ['/home/xww0766/solana/training_output/checkpoint-20000']
# Open the second file in read mode
with open('/home/lhhao0430/src_solana/datas/new_log_data/mix/test3.txt', 'r') as file:
    # Read all lines from the second file
    lines = file.readlines()

december_test_num = 1569
september_test_num = 1500
gt_labels = ['malicious'] * 18 + ['benign'] * december_test_num + ['benign'] * september_test_num
write_scores = False

# Hyper-parameter
g = 15
s = 4
max_seq_len = 4096*2
batch_size = 2

all_ma_score, all_september_score, all_december_score = [], [], []
all_ma_percentage, all_september_percentage, all_december_percentage = [], [], []
all_recall_rate, all_precision_rate = [[],[],[]], [[],[],[]]

for model_id, model_path in enumerate(tqdm(model_dirs)):
    config = AutoConfig.from_pretrained(model_path, **config_kwargs)
    model = AutoModelForMaskedLM.from_pretrained(
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
    scores, pad_nums = [], []

    for i in tqdm(range(0, len(lines), batch_size), desc="Batch Processing"):
        batch = lines[i:i+batch_size]

        # Tokenize the batch
        batch_encoding = tokenizer(batch, return_tensors="pt", padding='max_length', truncation=True, max_length=max_seq_len)
        input_ids = batch_encoding['input_ids']
        attention_masks = batch_encoding['attention_mask']
        
        pad_mask = torch.where(input_ids == 1, 1, 0)
        pad_num = torch.sum(pad_mask, dim=1)
        pad_nums.extend([num.item() for num in pad_num])

        #backup the input_ids before replacing
        input_ids_backup = input_ids.clone()
        masked_indices_batch = []
        
        # Randomly mask g% of tokens in each sequence
        for idx in range(input_ids.size(0)):
            seq_len = sum(attention_masks[idx]).item()  # Length of non-padded sequence
            masked_indices = np.random.choice(np.arange(1, seq_len), size=int(seq_len * g / 100), replace=False)

            for mi in masked_indices:
                input_ids[idx, mi] = tokenizer.mask_token_id
                
            masked_indices_batch.append(masked_indices)
                
        # Predict masked tokens
        input_tensor = input_ids.cuda()
        attention_tensor = attention_masks.cuda()
        logits = model(input_tensor, attention_mask=attention_tensor).logits
        
        # Process each sequence in the batch
        for idx in range(input_tensor.size(0)):
            abnormal_token_count = 0
            for mi in masked_indices_batch[idx]:
                probs = softmax(logits[idx, mi], dim=0)
                values, predictions = probs.topk(s)

                # Check for abnormal tokens
                if input_ids_backup[idx, mi] not in predictions:
                    abnormal_token_count += 1

            scores.append(abnormal_token_count)

    percentage=[]
    for i in range(len(scores)):
        percentage.append(scores[i]/((max_seq_len-pad_nums[i])*0.15))

    if write_scores:
        write_file("./scores" + str(model_id) + '.txt', scores)
        write_file("./percentage" + str(model_id) + '.txt', percentage)

        print("================")
        print(sum(scores[0:18])/18)
        print("ma_all_score:",scores[0:18])
        print(sum(percentage[0:18])/18)
        print("ma_all_percentage:",percentage[0:18])
        print(sum(pad_nums[0:18])/18)
        all_ma_score.append(sum(scores[0:18])/18)
        all_ma_percentage.append(sum(percentage[0:18])/18)

        print("================")
        print(sum(scores[18:december_test_num+18])/december_test_num)
        print(sum(percentage[18:december_test_num+18])/december_test_num)
        print(sum(pad_nums[18:december_test_num+18])/december_test_num)
        all_december_score.append(sum(scores[18:december_test_num+18])/december_test_num)
        all_december_percentage.append(sum(percentage[18:december_test_num+18])/december_test_num)
        
        print("================")
        print(sum(scores[december_test_num+18:])/september_test_num)
        print(sum(percentage[december_test_num+18:])/september_test_num)
        print(sum(pad_nums[december_test_num+18:])/september_test_num)
        all_september_score.append(sum(scores[december_test_num+18:])/september_test_num)
        all_september_percentage.append(sum(percentage[december_test_num+18:])/september_test_num)


        bins = np.linspace(0, 1, 50)
        plt.hist(percentage[0:18],bins=bins,alpha=0.5,label='Ma'+str(len(all_september_score)),density=True)

        plt.hist(percentage[18:18+december_test_num],bins=bins,alpha=0.5,label='Dec'+str(len(all_september_score)),density=True)

        plt.hist(percentage[18+december_test_num:],bins=bins,alpha=0.5,label='Sep'+str(len(all_september_score)),density=True)
        plt.xticks(rotation=90, fontsize=14)
        plt.legend()
        plt.savefig('data'+str(len(all_september_score))+'.png', bbox_inches='tight')
        plt.close()


    thresholds = [10, 15, 20]
    # Initialize lists to store metrics for each threshold
    fp_list, fpr_list = np.zeros(len(thresholds)), np.zeros(len(thresholds))
    precision_list, recall_list = np.zeros(len(thresholds)), np.zeros(len(thresholds))

    top_fp_list, top_fpr_list = np.zeros(3), np.zeros(3)
    top_precision_list, top_recall_list = np.zeros(3), np.zeros(3)

    success_list = np.zeros(len(thresholds))
    top_success_list = np.zeros(3)

    scores_np = np.array(percentage)
    sorted_indices = np.argsort(scores_np)[::-1]


    for threshold_index, threshold_value in enumerate(thresholds):

        predicted_labels = ['benign' for score in percentage]
        _len = min(threshold_value, len(percentage))
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

print("==============================================================")
if write_scores:
    print(all_ma_score)
    print(all_december_score)
    print(all_september_score)
    print(all_ma_percentage)
    print(all_december_percentage)
    print(all_september_percentage)
print(all_recall_rate[0])
print(all_recall_rate[1])
print(all_recall_rate[2])
print(all_precision_rate[0])
print(all_precision_rate[1])
print(all_precision_rate[2])
print("==============================================================")