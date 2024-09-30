import os
import json
import glob
import csv
import re
import copy
from tqdm import tqdm
from random import shuffle
from collections import Counter
# from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
# from transformers import BertTokenizerFast


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

def extract_function_calls(data, include_func_args=True, include_log_args=True, include_logs=True):
    # The concatenation format would be like [func sub_func-1 sub_log-1 sub_func-2 sub_log-2 log]
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
            traces.extend(extract_function_calls(call, include_func_args=True, include_log_args=True, include_logs=True))

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
    return traces


def write_all_traces(input_dir, config_dir, output_dir, remove_oov=True):
    top20k_address = read_txt(config_dir + '/top_20k_address.txt')
    top20k_hash = read_txt(config_dir + '/top_20k_hash.txt')

    reserverd_tokens = ['[CALL]', '[STATICCALL]', '[DELEGATECALL]', '[START]', '[END]', '[OOV]', '[INs]', '[OUTs]', '[STATE]', '[LOG]', '[NONE]', 'data', 'address'] + top20k_address + top20k_hash
    sub_dirs = os.listdir(input_dir)
    
    for sub_dir in sub_dirs:
        if sub_dir == '.DS_Store':
            continue
        all_traces, all_traces_path, num_traces = [], [], 0
        folder = input_dir + '/' + sub_dir + '/'
        # print (folder)
        json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
        json_files.sort()
        sub_traces, sub_traces_path = [], []
        for json_file in tqdm(json_files):
            # read json file
            t = json.load(open(folder + json_file))
            t_hash = list(t.keys())[-1]
            t = t[t_hash][0]
            # transform the original tx into a trace, ranked based on fuction calls 
            trace = extract_function_calls(t)
            # process the trace into pre-tokenization data
            for k, token in enumerate(trace):
                if is_address(token) or is_hash(token):
                    if token not in reserverd_tokens:
                        trace[k] = '[OOV]'
                elif is_decimal(token):
                    trace[k] = convert_decimal_to_hex_if_needed(token)
                elif is_rare_hex(token):
                    trace[k] = pad_hex(token)
                elif is_float(token):
                    token = int(token)
                    trace[k] = convert_decimal_to_hex_if_needed(token)
            sub_traces.append(copy.deepcopy(trace))
            sub_traces_path.append(folder + json_file)
            num_traces += 1
        all_traces.append(sub_traces)
        all_traces_path.append(sub_traces_path)

        output_folder = output_dir + '/' + sub_dir
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # save the pre-tokenization TXs 
        with open(output_folder + '/preprocessed_tx.txt', 'w') as file:
            for sub_traces in all_traces:
                train_num = int(len(sub_traces))
                for i in range(train_num):
                    trace = sub_traces[i]
                    trace_str =' '.join(map(str, trace))
                    if remove_oov:
                        # remove [OOV] to push the data within the distribution
                        trace_str = trace_str.replace(' [OOV]', '')
                    file.write(trace_str + '\n')

        # save the paths of the pre-tokenization TXs
        with open(output_folder + '/tx_path.txt', 'w') as file:
            for sub_traces_path in all_traces_path:
                num = int(len(sub_traces_path))
                for i in range(num):
                    trace_path = sub_traces_path[i]
                    file.write(trace_path + '\n')

if __name__ == '__main__':

    input_dir = '/home/xww0766/tx_fm/data/other_tests' # all the TXs that require tokenization
    config_dir = '/home/xww0766/tx_fm/config' 
    output_dir = '/home/xww0766/tx_fm/preprocessed_data' 

    if not os.path.exists(output_dir):
       os.makedirs(output_dir)
    write_all_traces(input_dir, config_dir, output_dir)
    # Running this file will generate the pre-tokenization data for all TXs saved in 'output_dir+/tokenized_tx.txt'
    # together with the path of the TXs saved in 'output_dir+/tx_path.txt'
    # 'output_dir+/tx_path.txt' has the same order as the 'output_dir+/tokenized_tx.txt' 
