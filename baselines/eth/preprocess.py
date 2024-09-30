import json
import os
import numpy as np
import requests
import os
from collections import Counter
import glob
import argparse
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizerFast
import requests
import math
import csv
import re
import pandas as pd
import copy
from random import shuffle
import random

csv.field_size_limit(2**24)

def hex_to_decimal(token):
    return int(token, 16)

def is_hex(token):
    try:
        int(token, 16)
        return True
    except:
        return False

def is_address(token):
    if type(token)!=str:
        return False
    return len(token) == 42 and token.startswith('0x')

def is_hash(token):
    if type(token)!=str:
        return False
    return len(token) == 10 and token.startswith('0x')

def is_rare_hex(token):
    if type(token)!=str:
        return False
    return 1 < len(token)< 64 and token.startswith('0x')

def decimal_to_hex_string(decimal_value, padding=64):
    hex_value = hex(decimal_value)[2:]  # Remove the '0x' prefix
    return f"0x{hex_value.zfill(padding)}"

def pad_hex(value, padding=64):
    hex_value = value[2:]  # Remove the '0x' prefix
    return f"0x{hex_value.ljust(padding, '0')}"

def is_decimal(value):
    if not isinstance(value, int):
        return False
    return re.fullmatch(r'\d+', str(value)) is not None

def is_float(value):
    return isinstance(value, float)

def round_to_significant_figures(num, n_sig_fig):
    if num == 0:
        return 0
    scale = int(-1 * math.floor(math.log10(abs(num))) + (n_sig_fig - 1))
    return round(num, scale)

def convert_decimal_to_hex_if_needed(value):
    if is_decimal(value):
        return decimal_to_hex_string(int(value))
    return value

def get_from_dict(data, name):
    ret = data.get(name, '[NONE]')
    if ret == None:
        ret = '[NONE]'
    return ret

def read_txt(file_name):
    keys_list = []
    with open(file_name, 'r') as file:
        for line in file:
            key = line.split(':')[0].strip()
            keys_list.append(key)
    return keys_list

def file_iterator(corpus_dir, file_ext="txt"):
    file_paths = glob.glob(corpus_dir + '/*.txt')
    print("Found {} files in {}".format(len(file_paths), corpus_dir))
    for file_path in file_paths[:20000]:
        with open(file_path, 'r') as f:
            content = f.read()
        yield content


def read_freq_file(address_file):
    addresses = []
    with open(address_file, "r") as f:
        # Skip the first line containing the number of files
        f.readline()

        # Read and process each line
        for line in f:
            address = line.split(" ")[1]
            address = address[:-1]  # Remove the trailing ':'
            addresses.append(address)

    # Print the extracted addresses
    print("The length of the freq address is: ", len(addresses))
    return addresses

def read_hash_file(hash_file):
    with open(hash_file, "r") as f:
        lines = f.readlines()

    hashes = []
    for line in lines[1:]:  # Skip the first line
        parts = line.split()
        hash_with_colon = parts[1]
        hash_without_colon = hash_with_colon[:-1]  # Remove the trailing ':'
        hashes.append(hash_without_colon)
    print("The length of the freq hash is: ", len(hashes))
    return hashes


# The concatenation format would be like [func sub_func-1 sub_log-1 sub_func-2 sub_log-2 log]
def extract_function_calls(data, include_func_args=True, include_log_args=True, include_state_args=True):
    traces = []
    if data == None:
        return traces
    root_call = ['[START]', f'[{data["type"]}]', data['from'], data['to'], get_from_dict(data,'func'), get_from_dict(data,'gas'), get_from_dict(data,'value')]
    if include_func_args:
        if 'args' in data and data['args'] != None:
            root_call.append('[INs]')
            for arg in data['args']:
                data_type = 'data'
                if arg['type'] == 'address':
                    data_type = 'address'
                root_call.extend([data_type, arg['data']])
        if 'output' in data and data['output'] != None:
            root_call.append('[OUTs]')
            for out in data['output']:
                root_call.extend([out['type'], out['data']])
    root_call.append('[END]')
    traces.extend(root_call)
    
    if 'calls' in data and data['calls'] != None:
        for call in data.get('calls', []):
            traces.extend(extract_function_calls(call, include_func_args=True, include_log_args=True, include_state_args=True))

    if include_log_args:
        if 'logs' in data and data['logs'] != None:
            for log in data.get('logs', []):
                log_trace = ['[START]', '[LOG]', log['address'], log['topics'][0]['data']]
                if include_log_args:
                    for topic in log['topics'][1:]:
                        log_trace.extend([topic['type'], topic['data']])
                    for data_item in log['data']:
                        log_trace.extend([data_item['type'], data_item['data']])
                log_trace.append('[END]')
                traces.extend(log_trace)

    if include_state_args:
        if 'state' in data and data['state'] != None:
            for state in data.get('state', []):
                state_trace = ['[START]', '[STATE]']
                state_trace.extend([state['type'], state['key'], state['value']])
                state_trace.append('[END]')
                traces.extend(state_trace)
    return traces

def fetch_top20k_address_hash(input_dir, output_dir):
    sub_dirs = os.listdir(input_dir)

    top90k_address, top6k_hash = {}, {}
    for sub_dir in sub_dirs:
        folder = input_dir + '/' + sub_dir + '/'
        json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
        for json_file in json_files:
            # read json file
            t = json.load(open(folder + json_file))
            t_hash = list(t.keys())[0]
            t = t[t_hash][0]
            trace = extract_function_calls(t)
            for token in trace:
                if is_address(token):
                    if token not in top90k_address:
                        top90k_address[token] = 1
                    else:
                        top90k_address[token] += 1
                if is_hash(token):
                    if token not in top6k_hash:
                        top6k_hash[token] = 1
                    else:
                        top6k_hash[token] += 1
    # sort the hash, address
    top90k_address = dict(sorted(top90k_address.items(), key=lambda item: item[1], reverse=True))
    top6k_hash = dict(sorted(top6k_hash.items(), key=lambda item: item[1], reverse=True))

    top90k_address = dict(list(top90k_address.items())[:90000])
    top6k_hash = dict(list(top6k_hash.items())[:6000])

    with open(output_dir + '/top90k_address.txt', 'w') as file:
        for key, value in top90k_address.items():
            file.write(f'{key}: {value}\n')

    with open(output_dir + '/top6k_hash.txt', 'w') as file:
        for key, value in top6k_hash.items():
            file.write(f'{key}: {value}\n')

def write_all_traces(input_dir, output_dir):
    top90k_address = read_txt(output_dir + '/top90k_address.txt')
    top6k_hash = read_txt(output_dir + '/top6k_hash.txt')

    reserverd_tokens = ['[CALL]', '[STATICCALL]', '[DELEGATECALL]', '[START]', '[END]', '[OOV]', '[INs]', '[OUTs]', '[STATE]', '[LOG]', '[NONE]', 'READ', 'WRITE', 'data', 'address'] + top90k_address + top6k_hash
    sub_dirs = os.listdir(input_dir)
    all_traces, all_traces_path, num_traces = [], [], 0

    # tmp_dir = 'tmp'
    # if not os.path.exists(output_dir + '/' + tmp_dir):
    #     os.makedirs(output_dir + '/' + tmp_dir)
    
    for sub_dir in sub_dirs:
        folder = input_dir + '/' + sub_dir + '/'
        json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
        json_files.sort()
        sub_traces, sub_traces_path = [], []
        for json_file in json_files:
            # read json file
            t = json.load(open(folder + json_file))
            t_hash = list(t.keys())[-1]
            # t = t[t_hash][0]
            t = t[t_hash][0]
            trace = extract_function_calls(t)
            for k, token in enumerate(trace):
                if is_address(token) or is_hash(token):
                    if token not in reserverd_tokens:
                        trace[k] = '[OOV]'
                elif is_decimal(token):
                    trace[k] = round_to_significant_figures(token, 2)
                elif is_rare_hex(token):
                    trace[k] = pad_hex(token)
                elif is_float(token):
                    token = int(token)
                    trace[k] = round_to_significant_figures(token, 2)
            sub_traces.append(copy.deepcopy(trace))
            sub_traces_path.append(folder + json_file)
            # write into tmp dir
            # for k, token in enumerate(trace):
            #     if token in reserverd_tokens:
            #         trace[k] = ''
            # file_path = output_dir + '/' + tmp_dir + '/' + str(num_traces) + '.txt'
            # with open(file_path, 'w') as file:
            #      file.write(' '.join(map(str, trace)) + '\n')
            # num_traces += 1
        all_traces.append(sub_traces)
        all_traces_path.append(sub_traces_path)
    
    with open(output_dir + '/benign_tst.txt', 'w') as file:
        for sub_traces in all_traces:
            # train_num = int(len(sub_traces) * 0.8)
            train_num = len(sub_traces)
            for i in range(train_num):
                trace = sub_traces[i]
                file.write(' '.join(map(str, trace)) + '\n')
                # trace_path = sub_traces_path[i]
                # file.write(trace_path + '\n')

def tokenize_function(input_dir, output_dir):
    top20k_address = read_txt(output_dir + '/top90k_address.txt')
    top20k_hash = read_txt(output_dir + '/top6k_hash.txt')

    tokenizer = Tokenizer(models.WordPiece(unl_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"] + ['[CALL]', '[STATICCALL]', '[DELEGATECALL]', '[START]', '[END]', '[OOV]', '[INs]', '[OUTs]', '[STATE]', '[LOG]', '[NONE]', 'READ', 'WRITE','data', 'address'] + top20k_address + top20k_hash
    trainer = trainers.WordPieceTrainer(vocab_size=50000, special_tokens=special_tokens)
    tokenizer.train_from_iterator(file_iterator(output_dir + '/tmp'), trainer=trainer)
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id),
        ],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    new_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
    token_dir = output_dir + '/50k_tokenizer'
    if not os.path.exists(token_dir):
       os.makedirs(token_dir)
    new_tokenizer.save_pretrained(token_dir)

if __name__ == '__main__':
    input_dir = '/home/xww0766/ood_transaction/data/benign_test'
    output_dir = '/home/xww0766/ood_transaction/baselines/data'
    if not os.path.exists(output_dir):
       os.makedirs(output_dir)
    #fetch_top20k_address_hash(input_dir, output_dir)
    # write_all_traces(input_dir, output_dir)
    tokenize_function(input_dir, output_dir)