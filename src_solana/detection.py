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
from src_solana.foundation_model  import replace_roberta_attn

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


def write_file(output_file_path, data):
    with open(output_file_path, 'w') as output_file:
        for value in data:
            output_file.write(f"{value}\n")


# Load tokenizer
tokenizer_kwargs = {
        "cache_dir": None,
        "use_fast": True,
        "revision": 'main',
        "use_auth_token": False,
}

token_path = './tokenizer'
tokenizer = AutoTokenizer.from_pretrained(token_path, **tokenizer_kwargs)

config_kwargs = {
        "cache_dir": None,
        "revision": 'main',
        "use_auth_token": False
}
replace_roberta_attn(True, True) # use flash-attention to acclerate the evaluation

# Open the pre-tokenization data
data_path = '/home/lhhao0430/src_solana/datas/new_log_data/mix/test3.txt'
with open(data_path, 'r') as file:
    lines = file.readlines()

# Hyper-parameter
'''
param g: if random masking, mask percentage 
param s: select top s predicted tokens
'''
g = 15
s = 4
max_seq_len = 4096*2
batch_size = 2

model_dirs = ['/home/xww0766/solana/training_output/checkpoint-20000'] # support a list of models

for model_id, model_path in enumerate(tqdm(model_dirs)):
    # Load model
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

    write_file("./scores" + str(model_id) + '.txt', scores)
    write_file("./percentage" + str(model_id) + '.txt', percentage)