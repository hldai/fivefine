from models import modelutils


def bert_batch_collect(device, type_id_dict, pad_token_id, examples):
    input_ids = [x['token_id_seq'] for x in examples]
    mask_idxs = [x['mask_idx'] for x in examples]
    token_type_ids_list = [x['token_type_ids'] for x in examples]
    input_ids, attn_mask = modelutils.pad_id_seqs(input_ids, device, pad_token_id)
    token_type_ids = None
    if len(token_type_ids_list) > 0 and token_type_ids_list[0] is not None:
        token_type_ids = modelutils.pad_seq_to_len(token_type_ids_list, input_ids.size()[1], device)
    labels_list = [x['labels'] for x in examples]
    type_ids_list = [[type_id_dict[t] for t in labels] for labels in labels_list]
    return {
        'input_ids': input_ids,
        'attn_mask': attn_mask,
        'token_type_ids': token_type_ids,
        'mask_idxs': mask_idxs,
        'type_ids_list': type_ids_list,
        'labels_list': labels_list
    }


def train_batch_list_iter(train_batch_loader, gacc_step):
    batch_list = list()
    for batch in train_batch_loader:
        batch_list.append(batch)
        if len(batch_list) == gacc_step:
            yield batch_list
            batch_list = list()


class IterExampleBatchLoader:
    def __init__(self, example_loader, batch_size, n_iter=-1, n_steps=-1, collect_fn=None):
        self.example_loader = example_loader
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_steps = n_steps
        self.collect_fn = collect_fn

    def __iter__(self):
        return self.next_batch()

    def next_batch(self):
        it = 0
        step = 0
        batch_examples = list()
        while it != self.n_iter and step != self.n_steps:
            example_iter = iter(self.example_loader)
            for i, example in enumerate(example_iter):
                batch_examples.append(example)
                if len(batch_examples) >= self.batch_size:
                    if self.collect_fn is not None:
                        yield self.collect_fn(batch_examples)
                    else:
                        yield batch_examples
                    batch_examples = list()

                    step += 1
                    if step == self.n_steps:
                        break
            if len(batch_examples) > 0:
                if self.collect_fn is not None:
                    yield self.collect_fn(batch_examples)
                else:
                    yield batch_examples
                batch_examples = list()
            it += 1
