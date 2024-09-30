from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizerFast
import os
import json
import glob

def prepare_special_tokens(frequency_report, top_n_values):
    # Special indicators used in the tokenize_transaction function
    special_indicators = [
        "[START]", "[END]", "[instruction begin]", "[instruction end]", "[TRANSAC_SEP]", 
        "[program]", "[programId]", "[accounts]", "[parsed]", "[parsed info]", "[logMessages]", "OOV"
    ]

    # Initialize a list with special indicators to ensure they are at the front
    special_tokens_list = special_indicators[:]

    # Use a set to track already added tokens to avoid duplicates
    added_tokens = set(special_indicators)

    # Iterate over categories in the frequency report
    for category, values in frequency_report.items():
        top_n = top_n_values.get(category, None)
        if top_n is None:
            # If top_n is None, add all tokens for this category
            for token, _ in values:
                if token not in added_tokens:
                    special_tokens_list.append(token)
                    added_tokens.add(token)
        else:
            # Otherwise, add only the top N frequent tokens
            for token, _ in values[:top_n]:
                if token not in added_tokens:
                    special_tokens_list.append(token)
                    added_tokens.add(token)

    return special_tokens_list

def train_tokenizer(special_tokens, corpus_path_list, save_path):
    tokenizer = Tokenizer(models.WordPiece(unl_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=30000, special_tokens=special_tokens)
    tokenizer.train(corpus_path_list, trainer=trainer)
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
    new_tokenizer.save_pretrained(save_path) 


frequency_file = '/home/lhhao0430/src_solana/datas/all_september_transaction_frequency_counter/frequency_counter.json'  
with open(frequency_file, 'r') as f:
    frequency_counter = json.load(f)
    frequency_counter.pop("logMessages")
    #print(frequency_counter["program"])
    
top_n_values = {
    "program": None,  # 'None' means all are included, no 'OOV'
    "programId": None,
    "accounts": 7000,
    "parsed_type": None,
    "parsed_info_type": None,
    "parsed_info_name": 2000,
    "parsed_info_data": 10000,
}

special_tokens = prepare_special_tokens(frequency_counter, top_n_values)
#print(special_tokens)
special_tokens = ["[CLS]", "[PAD]", "[SEP]", "[MASK]", "[UNK]"] + special_tokens
corpus_path_list = ['/home/lhhao0430/src_solana/datas/no_log_data/september/temp.txt']
save_path = './tokenizer_test_no_log_no_log'
train_tokenizer(special_tokens, corpus_path_list, save_path)