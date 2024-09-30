import warnings
from typing import Optional, Tuple

import torch
from torch import nn
import transformers
from xformers import ops as xops

import math
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding, apply_rotary_pos_emb
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids


group_size_ratio = 1/4
use_flash_attn = False 
use_full = True
rope_scaling = None


def convert_attention_mask(attention_mask, num_heads):
    bsz, _, _, q_len = attention_mask.size()
    attn_bias = attention_mask.expand(bsz, num_heads, q_len, q_len)
    return attn_bias

# Roberta Embedding
class Embeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
        # End copy
        self.padding_idx = config.pad_token_id

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
        
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        
        seq_length = input_shape[1]
        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

class SelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.max_position_embeddings = config.max_position_embeddings
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        self.norm_factor = self.attention_head_size**-0.5
        self.drop_prob = config.attention_probs_dropout_prob if self.training else 0.0
        
        self.is_decoder = config.is_decoder
        self._init_rope()

    def _init_rope(self):
        if rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.attention_head_size, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = rope_scaling["type"]
            scaling_factor = rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)


        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=self.norm_factor,
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        kv_seq_len = key_layer.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_layer, seq_len=kv_seq_len)
        position_ids = torch.arange(kv_seq_len, dtype=torch.long, device=hidden_states.device).view(1, -1)
        
        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids)
        query_layer, key_layer = query_layer.contiguous(), key_layer.contiguous()

        bsz, q_len, _ = hidden_states.size()
        group_size = int(q_len * group_size_ratio)
        if q_len % group_size > 0:
            raise ValueError("q_len %d should be divisible by group size %d." % (q_len, group_size))
        num_group = q_len // group_size
        
        if self.training and not use_full:
            def shift(qkv, num_heads, head_dim):
                # qkv = [bsz, nh, q_len, d]
                #   -> qkv = [bsz, q_len, nh, d]
                qkv = qkv.transpose(1, 2)
               
                qkv[:, :, num_heads//2:] = qkv[:, :, num_heads//2:].roll(-group_size//2, dims=1)

                # -> [bsz * n_group, group_s, nh, d)
                #   -> [bsz * n_group, nh, group_s, d)
                qkv = qkv.reshape(bsz * num_group, group_size, num_heads, head_dim).transpose(1, 2)
                return qkv
            def shift_mask(m, num_heads):
                # -> [bsz, nh, q_len, d]
                #   -> [bsz, q_len, nh, d]
                m = m.transpose(1, 2)
                m[:, :, num_heads//2:] = m[:, :, num_heads//2:].roll(-group_size//2, dims=1)
                # -> [bsz * n_group, group_s, nh)
                #   -> [bsz * n_group, nh, group_s)
                m = m.reshape(bsz * num_group, group_size, num_heads).transpose(1, 2)
                # -> [bsz * n_group, nh, group_s, group_s]
                m = m.unsqueeze(2).repeat(1, 1, group_size, 1)
                return m

            num_heads, head_dim = self.num_attention_heads, self.attention_head_size

            # contiguous is required as self._attn() will attempt to apply .view() on them
            query_layer = shift(query_layer, num_heads, head_dim).contiguous()
            key_layer = shift(key_layer, num_heads, head_dim).contiguous()
            value_layer = shift(value_layer, num_heads, head_dim).contiguous()

            # -> [bsz, 1, 1, q_len] 
            #     -> [bsz, 1, q_len] 
            #        -> [bsz, nh, q_len]
            attention_mask = attention_mask.squeeze(2)
            attention_mask = attention_mask.expand(bsz, num_heads, q_len).clone().detach()

            attention_mask = shift_mask(attention_mask, num_heads).contiguous()
            
        # Compute attention
        if use_flash_attn:
            # -> [bsz, nh, q_len, d]
            #   ->[bsz, q_len, nh, d]
            # flash attention require the input dimension should be [bsz, q_len, nh, d]
            query_layer = query_layer.transpose(1, 2).contiguous()
            key_layer = key_layer.transpose(1, 2).contiguous()
            value_layer = value_layer.transpose(1, 2).contiguous()
            
            if use_full:
                # -> [bsz, 1, 1, q_len]
                #   -> [bsz, nh, q_len, q_len]
                attn_bias = convert_attention_mask(attention_mask, self.num_attention_heads)
            else:
                assert self.training == True
                attn_bias = attention_mask
            # Flash attention implemented by Meta
            attn_bias = attn_bias.to(query_layer.dtype)
            attn_output = xops.memory_efficient_attention(query_layer, key_layer, value_layer, attn_bias=attn_bias, p=0.0)
        else:
            attn_output, attn_weights = self._attn(query_layer, key_layer, value_layer, attention_mask, head_mask)

        if not use_flash_attn:
           # -> [bsz, nh, q_len, d]
           #   ->[bsz, q_len, nh, d]
           attn_output = attn_output.transpose(1, 2).contiguous()

        # # NOTE: shift back
        if self.training and not use_full:
            attn_output = attn_output.reshape(bsz, q_len, self.num_attention_heads, self.attention_head_size)
            # [bsz, q_len, nh, hd]
            attn_output[:, :, self.num_attention_heads//2:] = attn_output[:, :, self.num_attention_heads//2:].roll(group_size//2, dims=1)

        attn_output = attn_output.reshape(bsz, q_len, self.all_head_size)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

def replace_roberta_attn(_use_flash_attn=True, _use_full=False, _rope_scaling=None):
    global use_flash_attn, use_full, rope_scaling
    use_flash_attn, use_full, rope_scaling = _use_flash_attn, _use_full, _rope_scaling

    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if _use_flash_attn and cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
            "Resorting to plain attention..."
        )
        use_flash_attn = False
    transformers.models.roberta.modeling_roberta.RobertaEmbeddings = Embeddings
    transformers.models.roberta.modeling_roberta.RobertaSelfAttention = SelfAttention