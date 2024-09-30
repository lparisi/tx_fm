import os
os.environ["CUDA_VISIBLE_DEVICES"]="7" ## Specify the GPU to use

import sys
sys.path.append('/home/xww0766/tx_fm/')

import json
import logging
import math
import shutil
import pandas as pd
import numpy as np

import csv
from tqdm import tqdm
from itertools import chain
from typing import Optional

from sklearn.svm import OneClassSVM
from sklearn import cluster
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
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

from src.foundation_model import replace_roberta_attn

np.random.seed(42)

# Load Model and tokenizer
tokenizer_kwargs = {
        "cache_dir": None,
        "use_fast": True,
        "revision": 'main',
        "use_auth_token": False,
}


token_path = './config/tokenizer' # this can be fixed
tokenizer = AutoTokenizer.from_pretrained(token_path, **tokenizer_kwargs)

config_kwargs = {
        "cache_dir": None,
        "revision": 'main',
        "use_auth_token": False
}

replace_roberta_attn(False, True)
model_path = './model/rope_roberta' # this can be fixed 

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


def squared_difference(a, b, do_normalization=True):
    """Computes (a-b) ** 2."""
    if do_normalization:
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 2. - 2. * np.dot(a, b.T)


def compute_hidden_representation(data, model, tokenizer, seq_len=1024):
    """
    :param data: a line in the .txt
    """

    model = model.cuda()
    model = model.eval()

    input_ids = tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)["input_ids"]
    input_ids = input_ids[0]

    # padding
    current_size = input_ids.size(0)
    remainder = current_size % 8
    padding_size = (8 - remainder) % 8
    input_ids = F.pad(input_ids, (0, padding_size), 'constant', 1)
    # padding_size = 0

    # compute <cls> representations
    input_tensor = torch.tensor([input_ids.tolist()])
    input_tensor = input_tensor.cuda()

    attention_mask = torch.ones((len(input_ids)))
    # if padding_size !=0:
    #     attention_mask[-padding_size:] = 0
    attention_mask = attention_mask.cuda().unsqueeze(0)

    # Extract the feature of the <CLS> token
    x = model.roberta(input_ids=input_tensor, attention_mask=attention_mask)[0]
    # use the cls token feature to cal the distance
    representation = x.detach()[:, 0, :] # this gives the cls token representation

    return representation.detach().cpu().numpy()


def find_nearest_farthest(test_data, test_contract_name, ref_set, model, tokenizer, num_tx=1, 
                          computed_ref_rep=None, load_ref_rep=False):
    """
    :param num_tx: return top num_tx nearest and farthest transactions in ref_set
    """
    # TODO: enable batch for ref_set
    
    ref_data = ref_set[0]
    ref_path = ref_set[1]

    ref_rep = []
    ref_pa = []
    
    if load_ref_rep:
        for rep, path in zip(computed_ref_rep, ref_path):
            if test_contract_name not in path and test_contract_name[0:5] not in path:
                continue
            ref_rep.append(rep)
            ref_pa.append(path)
    else:
        for data, path in zip(ref_data, ref_path):
            if test_contract_name not in path and test_contract_name[0:5] not in path:
                continue
            rep = compute_hidden_representation(data=data, model=model, tokenizer=tokenizer)
            ref_rep.append(rep)
            ref_pa.append(path)

    if len(ref_rep) == 0:
        print('the reference set does not have a tx coming from the same contract as the testing tx.')
        return 0

    ref_rep = np.array(ref_rep)
    if len(ref_rep.shape)==3:
        ref_rep = ref_rep[:, 0, :]

    test_rep = compute_hidden_representation(data=test_data, model=model, tokenizer=tokenizer)

    # Compute Euclidean distances between each test sample and all training samples
    distances = squared_difference(test_rep, ref_rep)

    nearest_path, farthest_path = [], []
    # Find nearest and farthest training samples for each test sample
    nearest_indices = np.argsort(distances, axis=1)[0, 0:num_tx]
    farthest_indices = np.argsort(distances, axis=1)[0, -num_tx:]

    nearest_path = [ref_pa[i] for i in nearest_indices]
    farthest_path = [ref_pa[i] for i in farthest_indices]

    return nearest_path, farthest_path


if __name__ == '__main__':

    # The result replicates the csv file shared previously
    # https://docs.google.com/spreadsheets/d/1XjbRyhdKbVCVR5hCD93cAZLqiAO5fetZ8CCp8Rzv9ks/edit?usp=sharing
    # https://docs.google.com/spreadsheets/d/1UcLq3AagB7ggrPy9MtyvXUf2z3USE8yb5DA7INGOw40/edit?usp=sharing
    
    data_dir = './preprocessed_data/malicious'
    # Open the pre-tokenization data 
    with open(data_dir+'/preprocessed_tx.txt', 'r') as file:
        lines = file.readlines()
        file.close()

    # read tx path
    with open(data_dir+'/tx_path.txt', 'r') as file:
        path_lines = file.readlines()
        file.close()

    ref_dir = './preprocessed_data/benign_training'

    with open(ref_dir+'/preprocessed_tx.txt', 'r') as file:
        ref_lines = file.readlines()
        file.close()

    with open(ref_dir+'/tx_path.txt', 'r') as file:
        ref_paths = file.readlines()
        file.close()

    for test_data, test_path in tqdm(zip(lines, path_lines)):
        test_contract_name = test_path.split('/')[-1].split('_')[0]
        near, far = find_nearest_farthest(test_data, test_contract_name, [ref_lines, ref_paths], fm_model, tokenizer)
        print('***************************************************')
        print ('the testing transaction:'+test_path)
        print ('the nearest transaction in the reference set:')
        print (near)
        print ('the farthest transaction in the reference set:')
        print (far)




# Write the results to a .csv file
# head = ["Test transactions", "Most similar", "Most dis-similar"]
# # Specify the CSV file path
# csv_file_path = "result.csv"
# # Write data to the CSV file line by line
# with open(csv_file_path, mode='w', newline='') as file:
#     # Create a CSV writer object
#     csv_writer = csv.writer(file)
#     # Write each line
#     csv_writer.writerow(head)
#     for m, n, f in zip(test_path, nearest_path, farthest_path):
#         m = "/".join(m.split('/')[-2:])
#         n = "/".join(n.split('/')[-2:])
#         f = "/".join(f.split('/')[-2:])
#         m = m.replace(" ", "").replace("\n", "")
#         n = n.replace(" ", "").replace("\n", "")
#         f = f.replace(" ", "").replace("\n", "")
#         csv_writer.writerow([m, n, f])