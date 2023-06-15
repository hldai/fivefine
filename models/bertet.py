import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from models.bertmlmcls import BertForMaskedLMHidden, BertOnlyMLMHead
from transformers.models.bert.configuration_bert import BertConfig


class Layer_Process(nn.Module):
    def __init__(self, process_sequence, hidden_size, dropout=0.1):
        super().__init__()
        self.process_sequence = process_sequence.lower()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        if 'd' in self.process_sequence:
            self.dropout = nn.Dropout(dropout)
        if 'n' in self.process_sequence:
            self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inp):
        output = inp
        for op in self.process_sequence:
            if op == 'a':
                output = output + inp
            elif op == 'd':
                output = self.dropout(output)
            elif op == 'n':
                output = self.layer_norm(output)
        return output


def seq_len_to_mask(seq_len, max_len=None):
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask


class MultiHeadAttentionDirect(nn.Module):
    def __init__(self, hidden_size, num_heads, scaled=True, attn_dropout=0.1,
                 post_dropout=0.1, ff_final=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        assert (self.per_head_size * self.num_heads == self.hidden_size)

        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_final = nn.Linear(self.hidden_size, self.hidden_size)
        self.u = nn.Parameter(torch.zeros(self.num_heads, self.per_head_size, requires_grad=True), requires_grad=True)
        self.v = nn.Parameter(torch.zeros(self.num_heads, self.per_head_size, requires_grad=True), requires_grad=True)

        self.dropout = nn.Dropout(attn_dropout)

        if ff_final:
            self.ff_final = nn.Linear(self.hidden_size, self.hidden_size)

        self.layer_postprocess = Layer_Process('an', self.hidden_size, post_dropout)

    def forward(self, key, query, value, seq_len, lex_num):
        key = self.w_k(key)
        value = self.w_v(value)

        batch = key.size(0)
        max_seq_len = key.size(1)
        q_batch_size = query.size()[0]

        key = torch.reshape(key, [batch, max_seq_len, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [q_batch_size, 1, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, max_seq_len, self.num_heads, self.per_head_size])

        # batch * n_head * seq_len * d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        # batch * n_head * d_head * key_len
        key = key.transpose(-1, -2)
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
        query_and_u_for_c = query + u_for_c
        attn_score_raw = torch.matmul(query_and_u_for_c, key)

        if self.scaled:
            attn_score_raw = attn_score_raw / math.sqrt(self.per_head_size)

        mask = seq_len_to_mask(seq_len + lex_num).bool().unsqueeze(1).unsqueeze(1)
        attn_score_raw_masked = attn_score_raw.masked_fill(~mask, -1e15)

        attn_score = F.softmax(attn_score_raw_masked, dim=-1)
        attn_score = self.dropout(attn_score)

        value_weighted_sum = torch.matmul(attn_score, value)

        result = value_weighted_sum.transpose(1, 2).contiguous().reshape(batch, 1, self.hidden_size)

        if hasattr(self, 'ff_final'):
            result = self.ff_final(result)
        result = result.view(batch, self.hidden_size)

        result = self.layer_postprocess(result)
        return result


class TypeTokenBertET(nn.Module):
    def __init__(self, bert_conf_dir, bert_dir=None):
        super(TypeTokenBertET, self).__init__()

        if bert_dir is None:
            bert_config = BertConfig.from_pretrained(bert_conf_dir)
            self.mlm = BertForMaskedLMHidden(bert_config)
        else:
            self.mlm = BertForMaskedLMHidden.from_pretrained(bert_dir)

        self.num_heads = 4
        self.token_weights = self.mlm.cls.predictions.decoder.weight
        weight_size = self.token_weights.data.size()
        self.h = weight_size[1]
        self.type_attn_layer = MultiHeadAttentionDirect(
            self.h, self.num_heads, scaled=True, ff_final=True)
        self.type_attn_embed = nn.Parameter(
            torch.randn((1, self.h), dtype=torch.float32, requires_grad=True) * 0.1,
            requires_grad=True)
        # self.cls_lin = nn.Linear(self.h, n_types)

        self.nw_cls = BertOnlyMLMHead(self.mlm.config)
        self.nw_cls.load_state_dict(self.mlm.cls.state_dict())
        # print(self.nw_cls.state_dict())
        # print(self.mlm.cls.state_dict())
        # exit()

        self.n_types = -1
        self.type_token_seq_lens = None
        self.type_token_ids_tensor = None

    def init_type_hiddens(self, tokenizer, type_vocab, device=None):
        from models import modelutils

        type_token_ids = list()
        for i, t in enumerate(type_vocab):
            t = t.replace('_', ' ')
            type_tokens = tokenizer.tokenize(t)
            type_token_ids.append((tokenizer.convert_tokens_to_ids(type_tokens)))

        if device is None:
            device = self.mlm.device
        self.n_types = len(type_vocab)
        # max_type_token_seq_len = max(len(seq) for seq in type_token_ids)
        self.type_token_seq_lens = torch.tensor([len(seq) for seq in type_token_ids], dtype=torch.int32, device=device)
        self.type_token_ids_tensor, type_token_mask = modelutils.pad_id_seqs(
            type_token_ids, device, tokenizer.pad_token_id)

    def forward(self, input_ids, attn_mask, mask_idxs, mode='tt', token_type_ids=None):
        cur_batch_size = input_ids.size()[0]
        if mode == 'nw':
            sequence_output = self.mlm(
                input_ids, attn_mask, token_type_ids=token_type_ids, return_sequence_output=True)
            _, all_nw_mlm_logits = self.nw_cls(sequence_output)
            nw_mlm_logits = all_nw_mlm_logits[np.arange(cur_batch_size), mask_idxs, :]
            return nw_mlm_logits

        vecs_type_tokens = self.token_weights[self.type_token_ids_tensor.view(-1)]
        vecs_type_tokens = vecs_type_tokens.view(self.n_types, -1, self.h)
        # print(vecs_type_tokens.size())

        self.type_hiddens = self.type_attn_layer(
            vecs_type_tokens, self.type_attn_embed, vecs_type_tokens, self.type_token_seq_lens, 0)
        self.type_hiddens = torch.t(self.type_hiddens)

        all_hidden_vecs, mlm_logits = self.mlm(input_ids, attn_mask, token_type_ids=token_type_ids)

        hidden_vecs = all_hidden_vecs[np.arange(cur_batch_size), mask_idxs, :]
        logits = torch.matmul(hidden_vecs, self.type_hiddens)

        return logits, mlm_logits

    @staticmethod
    def from_trained(model_file, bert_model='bert-base-cased'):
        state_dict_tmp = torch.load(model_file, map_location='cpu')
        state_dict = dict()
        for k, v in state_dict_tmp.items():
            if k.startswith('module'):
                state_dict[k[7:]] = v
            else:
                state_dict[k] = v

        del_keys = list()
        for k, v in state_dict.items():
            if k.startswith('cls_lin.'):
                del_keys.append(k)
        for k in del_keys:
            del state_dict[k]

        model = TypeTokenBertET(bert_conf_dir=bert_model)
        model.load_state_dict(state_dict, strict=False)
        return model
