# -->
# 	Add a summary function.
# -->

import os
os.environ["CUDA_VISIBLE_DEVICES"]="7" ## Specify the GPU to use

import sys
sys.path.append('/home/xww0766/tx_fm/')
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.functional import softmax
import torch.nn.functional as F

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
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


def exp_TX(data_dir, model=fm_model, tokenizer=tokenizer, batch_size=16, use_prob=True,
           exp_method='grad', g=15, s=3, seq_len=1024):
    """
    :param data_dir: path to the tokenized data of the testing TXs (The folder should has a set of )
    :param batch_size: batch size of computing explanation (TODO: include later)
    :param use_prob: when deriving the gradient (dy/dx), use the probability (pi) or the loss (-log pi)
    :param exp_method: explanation method (i.e., grad, integratedgrad, smoothgrad)
    :param g: mask percentage
    :param s: top_k
    """

    def get_loss_token(input, attn_mask, g, s):
        """
        Input with size (seq_len,)
        :param m_size: number of tokens masked in each masked input
        """
        loss_tuple = []
        masked_tokens = []
        masked_indices = np.random.choice(np.arange(1, len(input)), size=int(len(input) * g / 100), replace=False)
        for idx in masked_indices:
            masked_tokens.append((idx, input[idx].item()))
            input[idx] = tokenizer.mask_token_id

        # Predict masked tokens
        input_tensor = torch.tensor([input.tolist()])
        input_tensor = input_tensor.cuda()
        logits = model(input_tensor, attn_mask).logits

        # Count abnormal tokens
        for idx, original_token_id in masked_tokens:
            probs = softmax(logits[0, idx], dim=0)
            loss = -1.0 * torch.log(probs[original_token_id]).item()
            values, predictions = probs.topk(s) # select top s predicted tokens

            if original_token_id not in predictions:
                loss_tuple.append((loss, idx, 0))
            else:
                loss_tuple.append((loss, idx, 1))

        # Sort the loss_tuple based on the loss value in descending order
        loss_tuple = sorted(loss_tuple, key=lambda x: -x[0])
        return loss_tuple

    # Open the pre-tokenization data
    with open(data_dir + '/preprocessed_tx.txt', 'r') as file:
        lines = file.readlines()

    # read tx path
    with open(data_dir + '/tx_path.txt', 'r') as file:
        path_lines = file.readlines()

    model = model.cuda()
    model = model.eval()

    grad_explainer = GradientExp(model, use_prob)

    ret_exp = {}

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

        attention_mask = torch.ones((len(input_ids)))
        if padding_size != 0:
            attention_mask[-padding_size:] = 0
        attention_mask = attention_mask.cuda().unsqueeze(0)

        loss_tuple = get_loss_token(torch.clone(input_ids), attention_mask, g, s)

        # Get the overall explanation
        exp = np.zeros_like(input_ids)

        for i in tqdm(range(len(loss_tuple))):
            _, idx, correct = loss_tuple[i]
            if correct == 1:
                continue
            original_token_id = input_ids[idx].item()
            if original_token_id == 2 or original_token_id == 3:
                continue
            input_ids[idx] = tokenizer.mask_token_id
            input_tensor = torch.tensor([input_ids.tolist()])
            input_tensor = input_tensor.cuda()

            x = model.roberta.embeddings(input_ids=input_tensor)
            x = x.detach()
            # refer to line 941 in "https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py"
            attention_mask_1 = attention_mask[:, None, None, :]
            dtype = attention_mask.dtype
            attention_mask_reverse = (1 - attention_mask_1) * torch.finfo(
                dtype).min  # added to the input to rule out padding tokens

            if exp_method == 'grad':
                exp_x = grad_explainer.grad(x, attention_mask_reverse, idx, original_token_id)
            elif exp_method == 'integratedgrad':
                exp_x = grad_explainer.integratedgrad(x, attention_mask_reverse, idx, original_token_id)
            else:
                exp_x = grad_explainer.smoothgrad(x, attention_mask_reverse, idx, original_token_id)

            input_ids[idx] = original_token_id
            exp = exp_x + exp

        exp_max = np.max(exp)
        exp_max = 1e-6 if exp_max == 0.0 else exp_max
        exp /= exp_max

        _, ret_tx = get_token_score(data, exp, 0, tokenizer, input_ids)
        ret_exp[name] = (ret_tx)

    return ret_exp


def exp_TX_top_k(data_dir,  model=fm_model, tokenizer=tokenizer, use_prob=True, exp_method='grad',
           top_k=3, seq_len=1024):

    """
    :param data_dir: path to the tokenized data of the testing TXs (The folder should has a set of )
    :param model: model to use
    :param output_top_k: output the top k wrongly predicted tokens
    :param use_prob: when deriving the gradient (dy/dx), use the probability (pi) or the loss (-log pi)
    :param exp_method: explanation method (i.e., grad, integratedgrad, smoothgrad)
    :param top_k: for each token within the transactions (txs), we retrieve the token associated with the top-k highest prediction loss
    """

    def get_loss_token(input, attn_mask, m_size=16, batch_size=4):
        """
        Input with size (seq_len,)
        :param m_size: number of tokens masked in each masked input
        """
        loss_tuple = []
        assert seq_len % m_size == 0
        num_samples = seq_len // m_size
        assert num_samples % batch_size == 0
        n_batch = num_samples // batch_size

        index_list = list(range(0, 1024))
        np.random.shuffle(index_list)
        attn_mask = torch.stack([attn_mask] * batch_size)

        # a = 0
        # b = 0
        count = 0
        for _ in range(n_batch):
            input_batch = torch.stack([input] * batch_size)
            masked_ids_batch, original_tokens_batch = [], []
            for i in range(batch_size):
                masked_ids = index_list[count*m_size:(count+1)*m_size]
                original_tokens = input[masked_ids]
                input_batch[i, masked_ids] = tokenizer.mask_token_id
                count += 1
                masked_ids_batch.append(masked_ids)
                original_tokens_batch.append(original_tokens)
            
            input_batch = input_batch.cuda()
            logits = model(input_ids=input_batch, attention_mask=attn_mask).logits

            for j in range(batch_size):
                for idx, original_token_id in zip(masked_ids_batch[j], original_tokens_batch[j]):
                    # b = b + 1
                    probs = softmax(logits[j, idx], dim=0)
                    loss = -1.0 * torch.log(probs[original_token_id]).item()
                    values, predictions = probs.topk(3)

                    if original_token_id not in predictions:
                        loss_tuple.append((loss, idx, 0))
                        # a = a+1
                    else:
                        loss_tuple.append((loss, idx, 1))

        # Sort the loss_tuple based on the loss value in descending order
        loss_tuple = sorted(loss_tuple, key=lambda x: -x[0])
        return loss_tuple

    # Open the pre-tokenization data 
    with open(data_dir+'/preprocessed_tx.txt', 'r') as file:
        lines = file.readlines()

    # read tx path
    with open(data_dir+'/tx_path.txt', 'r') as file:
        path_lines = file.readlines()

    model = model.cuda()
    model = model.eval()

    grad_explainer = GradientExp(model, use_prob)

    ret_exp = {}

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

        attention_mask = torch.ones((len(input_ids)))
        if padding_size != 0:
            attention_mask[-padding_size:] = 0
        attention_mask = attention_mask.cuda().unsqueeze(0)

        loss_tuple = get_loss_token(torch.clone(input_ids), attention_mask)

        exp_top_k = []
        for i in range(top_k):
            _, idx, _ = loss_tuple[i]
            original_token_id = input_ids[idx].item()
            if original_token_id==2 or original_token_id==3:
                continue
            input_ids[idx] = tokenizer.mask_token_id
            input_tensor = torch.tensor([input_ids.tolist()])
            input_tensor = input_tensor.cuda()

            x = model.roberta.embeddings(input_ids=input_tensor)
            x = x.detach()
            # refer to line 941 in "https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py"
            attention_mask_1 = attention_mask[:, None, None, :]
            dtype = attention_mask.dtype
            attention_mask_reverse = (1 - attention_mask_1) * torch.finfo(dtype).min # added to the input to rule out padding tokens

            if exp_method == 'grad':
                exp_x = grad_explainer.grad(x, attention_mask_reverse, idx, original_token_id)
            elif exp_method == 'integratedgrad':
                exp_x = grad_explainer.integratedgrad(x, attention_mask_reverse, idx, original_token_id)
            else:
                exp_x = grad_explainer.smoothgrad(x, attention_mask_reverse, idx, original_token_id)

            input_ids[idx] = original_token_id

            ret_idx, ret_tx = get_token_score(data, exp_x, idx, tokenizer, input_ids)

            # get the actual token of the idx
            exp_top_k.append((ret_idx, ret_tx))
        ret_exp[name] = exp_top_k

    return ret_exp


def get_token_score(data, exp, idx, tokenizer, input_ids):
    # input_ids is for debugging prupose
    data = data.rstrip()
    words = data.split(' ')
    # encoding per word, has a longer length than input_ids
    enc = [tokenizer.encode(word, add_special_tokens=False) for word in words if len(word)!=0]
    ret_idx, ret_tx, st = {}, {}, 1 # skip the CLS token

    for i, token in enumerate(enc):
        if len(token) == 0:
            print('00000')
        ed = st + len(token)
        if ed > 1024:
           ed = 1024
        exp_token = sum(exp[st:ed]) / (ed - st)
        ret_tx[str(i) + '_' + words[i]] = exp_token
        if idx >= st and idx < ed:
            ret_idx[i] = words[i]
        st = ed
        if st >= 1024:
            break

    return ret_idx, ret_tx

class GradientExp(object):
    def __init__(self, model, use_prob=True):
        self.model = model
        self.use_prob = use_prob

    def class_gradients(self, x, attention_mask, idx, token):
        x.requires_grad = True
        encoder_output = self.model.roberta.encoder(x, attention_mask=attention_mask)
        sequence_output = encoder_output[0]
        logits = self.model.lm_head(sequence_output)
        probs = softmax(logits[0, idx], dim=0)

        y = probs[token]
        if not self.use_prob:
            y = -1.0 * torch.log(y)
        y.backward() # compute the gradient
        sal_x = x.grad.detach().cpu().numpy()
        return sal_x

    def grad(self, x, attention_mask, idx, token, normalize=True):

        sal_x = self.class_gradients(x, attention_mask, idx, token)
        sal_x = np.sum(sal_x, axis=-1).reshape(-1)
        if normalize:
            sal_x = np.abs(sal_x)
            sal_x_max = np.max(sal_x)
            sal_x_max = 1e-6 if sal_x_max == 0.0 else sal_x_max
            sal_x /= sal_x_max

        return sal_x

    def integratedgrad(self, x, attention_mask, idx, token, x_baseline=None, x_steps=25, normalize=True):
        if x_baseline is None:
            x_baseline = torch.zeros_like(x).cuda()
        else:
            assert x_baseline.shape == x.shape

        x_diff = x - x_baseline
        total_gradients = np.zeros_like(x.detach().cpu().numpy())

        for alpha in np.linspace(0, 1, x_steps):
            x_step = x_baseline + alpha * x_diff

            grads = self.class_gradients(x_step, attention_mask, idx, token)

            total_gradients += grads

        x_diff = x_diff.detach().cpu().numpy()
        sal_x = total_gradients * x_diff
        sal_x = np.sum(sal_x, axis=-1).reshape(-1)

        if normalize:
            sal_x = np.abs(sal_x)
            sal_x_max = np.max(sal_x)
            sal_x_max = 1e-6 if sal_x_max == 0.0 else sal_x_max
            sal_x /= sal_x_max

        return sal_x

    def smoothgrad(self, x, attention_mask, idx, token, stdev_spread=.15, nsamples=25, magnitude=True, normalize=True):

        stdev = stdev_spread * (torch.max(x).item() - torch.min(x).item())
        total_gradients = np.zeros_like(x.detach().cpu().numpy())

        for i in range(nsamples):
            noise = torch.normal(0, stdev, x.shape).cuda()
            x_plus_noise = x + noise
            grads = self.class_gradients(x_plus_noise, attention_mask, idx, token)
            if magnitude:
                total_gradients += (grads * grads)
            else:
                total_gradients += grads

        sal_x = total_gradients / nsamples
        sal_x = np.sum(sal_x, axis=-1).reshape(-1)

        if normalize:
            sal_x = np.abs(sal_x)
            sal_x_max = np.max(sal_x)
            sal_x_max = 1e-6 if sal_x_max == 0.0 else sal_x_max
            sal_x /= sal_x_max

        return sal_x

if __name__ == '__main__':
   data_dir = '../preprocessed_data/exp_test'
   exp = exp_TX(data_dir) # give the overall explanation of each token, where each token has a score between 0 and 1
   exp_top_k = exp_TX_top_k(data_dir)
   print('done')