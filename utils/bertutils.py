import torch


def get_bert_adam_optim(named_params, learning_rate, w_decay):
    # from transformers.optimization import AdamW
    from torch.optim import AdamW

    if w_decay == 0:
        # return AdamW([p for _, p in named_params], lr=learning_rate, correct_bias=False)
        return AdamW([p for _, p in named_params], lr=learning_rate)

    no_decay = ['bias', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': w_decay},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    assert len(optimizer_grouped_parameters[1]['params']) != 0
    # optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, weight_decay=w_decay, correct_bias=False)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, weight_decay=w_decay)
    return optimizer


def add_mlm_to_bert_input(input_ids, tokenizer, device, mlm_probability=0.15, rand_word_rate=0.1):
    mlm_labels = input_ids.clone()
    probability_matrix = torch.full(input_ids.shape, mlm_probability, device=device)
    special_tokens_mask = torch.tensor([
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in mlm_labels
    ], dtype=torch.bool, device=device)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    mlm_labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(
        torch.full(mlm_labels.shape, 0.8, device=device)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(
        torch.full(mlm_labels.shape, rand_word_rate, device=device)
    ).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), mlm_labels.shape, dtype=torch.long, device=device)
    input_ids[indices_random] = random_words[indices_random]
    return input_ids, mlm_labels
