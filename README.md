
## Code structure

- `./src-eth/preprocess.py`: preprocess the original transactions into pre-tokenization data (ready for tokenization)
- `./src-eth/foundation_model.py`: our fundation model (rope embedding + roberta)
- `./src-eth/detection.py`: take as input the testing TXs and run MLM on the foundation model for anomaly detection
- `./src-eth/post_analysis.py`: take as input a testing TX and a set of reference TXs, find the nearest and farthest TXs of the testing one in the reference set 
- `./src-eth/exp.py`: take as input a testing TX and its detection result, find the most important tokens
- `./data`: save the testing TXs and the reference TXs (the current data contains some examples, including our top TP, FP and some training ones for computing nearest and farthest)
- `./model`: save the foundation model; the current model is trained use the data in: https://drive.google.com/file/d/1hDPaakv1srExHb7S2FOxKbj7rUrtOIrN/view?usp=drive_link (20240125)
- `./config`: necessary files for tokenization

- `src-solana`: The sonala training project
  - `./src-solana/preprocess.py`: preprocess the original transactions into pre-tokenization data (ready for tokenization)
  - `./src-solana/detection.py`: take as input the testing TXs and run MLM on the foundation model for anomaly detection
  - `./src-solana/datapreprocess_fast.py`: multiprocess version of `./src-solana/preprocess.py`
  - `./src-solana/solana_pretrain.py`: pre-train the MLM on the preprocessed TXs
  - `./src-solana/foundation_model.py`: our fundation model (rope embedding + roberta)

- The pretrained model and tokenizer can be found at: https://drive.google.com/drive/folders/1JO1ZG-o4miWjCMNIMVVYUp9ndSH39__m?usp=drive_link 

## Workflow
- `preprocess.py` takes as input the .json files in the sub-folders of `./data` and put the preprocessed data (.txt files) in the `./preprocessed_data` folder with the same sub-folder name
- `detection.py` takes as input a folder of preprocessed data and run abnormal detection. It will also take as input a detection threshold. It will return the abnormal scores for all txs and the detected abnormal ones  
- `post_analysis.py` finds the nearest and farthest TXs from a set of reference TXs for one TX
- `exp.py` finds the most important tokens for a testing TX's detection result

## Explanation procedure
- Recall that our method pinpoint top malicious TXs based on how many tokens are wrongly predicted 
- For each transaction, we first find the top K tokens with the highest prediction loss, indicating these tokens are highly likely to carry malicious meanings. We can denote these tokens as malicious tokens 
- Then, for each malicious token, we assign each token in the input an importance score, indicating how important that token is to the model’s prediction for the malicious token.
- As such, we will have K group of importances for each token in the input (.txt file)
