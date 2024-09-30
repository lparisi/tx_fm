import json
import glob
from collections import defaultdict
import argparse
import os
from tqdm import tqdm
import time
import multiprocessing
import random

def is_numeric(s):
    """Check if a string represents a numerical value."""
    try:
        float(s)
        return True
    except:
        return False

def preprocess_instruction(instruction):
    """Convert None data values to 'null'."""
    if "parsed" in instruction and "info" in instruction["parsed"] and instruction["parsed"]["info"]!=None:
        for name, info in instruction["parsed"]["info"].items():
            if "data" in info and info["data"] is None:
                info["data"] = 'null'

def load_json_files(file_paths):
    """Load transactions from JSON files."""
    #print(file_paths)
    transaction_len=[]
    transactions1=[]
    transactions2 = []
    transaction_jsons=[]
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            transactions1.extend(data if isinstance(data, list) else [data])
            transactions2.append(data if isinstance(data, list) else [data])
            transaction_len.append(len(data))
            transaction_jsons.append(file_path)
            #print(len(data))
    return transactions1,transactions2,transaction_len,transaction_jsons

def analyze_instruction(instruction, counters):
    """Analyze an instruction and update frequency counters for each category."""
    preprocess_instruction(instruction)  # Preprocess to convert None to 'null'
    if "program" in instruction:
        counters["program"][instruction["program"]] += 1
    if "programId" in instruction:
        counters["programId"][instruction["programId"]] += 1
    if "accounts" in instruction:
        for account in instruction["accounts"]:
            account=account["pubkey"]
            counters["accounts"][account] += 1
    if "parsed" in instruction:
        parsed = instruction["parsed"]
        if "type" in parsed:
            counters["parsed_type"][parsed["type"]] += 1
        if ("info" in parsed) and (parsed["info"]!=None):
            for name, info in parsed["info"].items():
                counters["parsed_info_name"][name] += 1
                if "data" in info and not is_numeric(info["data"]):
                    counters["parsed_info_data"][info["data"]] += 1
                if "type" in info:
                    counters["parsed_info_type"][info["type"]] += 1

def analyze_transactions(transactions):
    """Analyze transactions and maintain separate frequency counters for each category."""
    counters = {
        "program": defaultdict(int),
        "programId": defaultdict(int),
        "accounts": defaultdict(int),
        "parsed_type": defaultdict(int),
        "parsed_info_name": defaultdict(int),
        "parsed_info_type": defaultdict(int),
        "parsed_info_data": defaultdict(int),
    }
    
    for transaction in transactions:
        #if 'tx' in transaction:
        for instruction in transaction['instructions']:
            analyze_instruction(instruction, counters)
    
    # Sort each category by frequency in descending order
    sorted_counters = {category: sorted(counts.items(), key=lambda x: x[1], reverse=True)
                       for category, counts in counters.items()}
    return sorted_counters

    
def is_top_frequent(token, category, frequency_report, top_n_values):
    if category not in frequency_report or top_n_values[category] is None:
        return True  # If the category is not limited, treat as always frequent

    top_tokens = set(item[0] for item in frequency_report[category][:top_n_values[category]])
    return token in top_tokens


def count_items(data):
    count = len(data)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, dict):
                count += count_items(value)
            if isinstance(value, list):
                count += count_items(value)
    if isinstance(data, list):
        for value in data:
            if isinstance(value, dict):
                count += count_items(value)
            if isinstance(value, list):
                count += count_items(value)
    return count

def tokenize_transaction(transaction, frequency_report, top_n_values):
    #print(count_items(transaction))    
    #time.sleep(10000)
    #print("++++++++++++++++++++++++++++++++++++")
    instruction_tokens = ["[START]"]
    log_messages_tokens = []

    for instruction in transaction.get('instructions', []):
        instruction_tokens.append("[instruction begin]")

        # Handle each category, ensuring missing ones result in 'null'
        for category in ["program", "programId", "accounts"]:
            tokens = instruction.get(category)
            instruction_tokens.append(f"[{category}]")
            if tokens:
                if isinstance(tokens, list):
                    for token in tokens:
                        if isinstance(token, dict):
                            token=token["pubkey"]
                        if is_top_frequent(token, category, frequency_report, top_n_values):
                            instruction_tokens.append(token)
                        else:
                            instruction_tokens.append("OOV")
                else:  # For non-list items, directly check frequency or mark as OOV
                    if is_top_frequent(tokens, category, frequency_report, top_n_values):
                        instruction_tokens.append(tokens)
                    else:
                        instruction_tokens.append("OOV")
            else:  # Property does not exist
                instruction_tokens.append("null")

        # Special handling for 'parsed'
        instruction_tokens.append("[parsed]")
        parsed = instruction.get("parsed")
        if parsed and not isinstance(parsed, str):
            parsed_type = parsed.get("type", "null")
            if "type" in parsed and is_top_frequent(parsed_type, "parsed_type", frequency_report, top_n_values):
                instruction_tokens.append(parsed_type)
            else:
                instruction_tokens.append("OOV")

            instruction_tokens.append("[parsed info]")
            #print(parsed.get("info", {}))
            if parsed.get("info", {})==None:
                #print("!!!")
                parsed["info"]={}
            
            
            for name, detail in parsed.get("info", {}).items():
                if is_top_frequent(name, "parsed_info_name", frequency_report, top_n_values):
                    instruction_tokens.append(name)
                else:
                    instruction_tokens.append("OOV")
                #print(detail)
                if detail==None:
                    #print("@@@@@@@@@@@")
                    detail={}
                detail_type = detail.get("type", "null")
                if is_top_frequent(detail_type, "parsed_info_type", frequency_report, top_n_values):
                    instruction_tokens.append(detail_type)
                else:
                    instruction_tokens.append("OOV")

                data_token = detail.get("data", "null")
                if isinstance(data_token, str) and is_numeric(data_token) == False and not is_top_frequent(data_token, "parsed_info_data", frequency_report, top_n_values):
                    data_token = "OOV"
                instruction_tokens.append(data_token)
        else:
            # Handle the case where 'parsed' is missing or a string
            instruction_tokens.append("null")

        # Special handling for 'logMessages'
        instruction_tokens.append("[logMessages]")
        log_messages = instruction.get("logMessages", ["null"])
        for log_message in log_messages:
            # Replace newline characters with a special string
            cleaned_log_message = log_message.replace("\n", "<nl>")
            instruction_tokens.append(cleaned_log_message)
            log_messages_tokens.append(cleaned_log_message)

        instruction_tokens.append("[instruction end]")

    instruction_tokens.append("[END]")
    #print("===",len(instruction_tokens)) 
    return instruction_tokens,log_messages_tokens

def setcallback(all_data):
    with open(all_data[0], 'a+') as output_file:
        for i in all_data[1]:
            output_file.write(i)
    with open(all_data[2], 'a+') as output_file:
        for i in all_data[4]:
            output_file.write(i)
        for i in all_data[3]:
            output_file.write(i)
        

def tokenizer_file(window_range,token_max_for_each_transaction,seq_max_len,transaction_len,output_file_path, transactions, frequency_report, top_n_values, transaction_jsons, write_temp=False, temp_file=None):
    
    tokenized_transactions = []
    transaction_lines = []
    numerical_values_lines = []
    log_messages_lines = []

    for i in range(transaction_len[1]):
        tokenized_transaction, log_messages_tokens = tokenize_transaction(transactions[i], frequency_report, top_n_values)
        tokenized_transactions.append(tokenized_transaction[:token_max_for_each_transaction])
        log_messages_lines.append(" ".join(log_messages_tokens) + "\n")
    for i in range(0, transaction_len[1], window_range):
        concat_transaction=[]
        flag=0
        for j in range(i,transaction_len[1]):
            if (len(concat_transaction)+len(tokenized_transactions[j])+1<=seq_max_len):
                concat_transaction.extend(tokenized_transactions[j])
                concat_transaction.append("[TRANSAC_SEP]")
            else:
                flag=1
                break
                
        concat_transaction.pop()
                
        # Prepare a list to hold only numerical values if write_temp is True
        numerical_values = []
        #print(len(concat_transaction))
        for token in concat_transaction:
            # Check if the token is a numerical value
            if is_numeric(token):
                numerical_values.append(str(token))
            # Convert all tokens to string for the main tokenized transaction
            token = str(token)
            
        # Write the main tokenized transaction
        transaction_line = " ".join([str(token) for token in concat_transaction]) + "\n"
        transaction_lines.append(transaction_line)
        numerical_values_line=" ".join(numerical_values) + "\n"      
        numerical_values_lines.append(numerical_values_line)
                    
        if flag==0:
            break
    print(transaction_len[0]," ",os.getpid())
    
    return [output_file_path,transaction_lines,temp_file,numerical_values_lines,log_messages_lines]
    
def fast_func():
    pass

if __name__ == '__main__':
    
    #pool = multiprocessing.Pool(6000)


    # Assuming you have a list of file paths
    file_paths1 = glob.glob('./../../../../sharedata/october/october_programs_filtered/*.json')  
    file_paths2 = glob.glob('./../../../../sharedata/solana_multi_transaction_data/september_programs_filtered/*.json')  
    file_paths3 = glob.glob('./../../../../sharedata/solana_multi_transaction_data/december_programs_filtered/*.json')  
    file_paths=file_paths1+file_paths2+file_paths3

    transactions1,transactions2,transaction_len,transaction_jsons= load_json_files(file_paths)
    frequency_file = '/home/lhhao0430/src_solana/datas/all_september_transaction_frequency_counter/frequency_counter.json'  
    if not os.path.exists(frequency_file):
        frequency_counter = analyze_transactions(transactions1)
        with open(frequency_file, 'w') as f:
            json.dump(frequency_counter, f, indent=4)
    else:
        with open(frequency_file, 'r') as f:
            frequency_counter = json.load(f)
    
    '''
    ##### Analysis results ####
    program: 62 unique strings (all)
    programId: 194 unique strings (all)
    accounts: 56,203 unique strings (7,000)
    parsed_type: 218 unique strings (all)
    parsed_info_name: 2260 unique strings (2,000)
    parsed_info_type: 5 unique strings (all)
    parsed_info_data: 112,377 unique strings (10,000)
    log_messages: 496,063 unique strings (8,000)
    '''
    
    top_n_values = {
    "program": None,  # 'None' means all are included, no 'OOV'
    "programId": None,
    "accounts": 7000,
    "parsed_type": None,
    "parsed_info_type": None,
    "parsed_info_name": 2000,
    "parsed_info_data": 10000,
    "logMessages": 8000,
    }
    
    output_file_path = '/home/lhhao0430/src_solana/datas/sep2dec_data_no_log_3_4096_mix/train_mix.txt'
    temp_file = '/home/lhhao0430/src_solana/datas/sep2dec_data_no_log_3_4096_mix/temp_mix.txt'
    
    if os.path.exists(temp_file):
        write_temp = False
        temp_file = None
    else:
        write_temp = True
    T1 = time.time()   
    pool = multiprocessing.Pool(8)
    for i in tqdm(range(len(transaction_len))):
        pool.apply_async(func=tokenizer_file, args=(9999999,512,3*4096,(i,transaction_len[i]),output_file_path, transactions2[i], frequency_counter, top_n_values, transaction_jsons[i], write_temp, temp_file),callback=setcallback)
    pool.close()
    pool.join()
    T2 = time.time() 
    print('time:%sms' % ((T2 - T1)*1000))