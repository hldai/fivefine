import numpy as np
from torch.utils.data import IterableDataset
from dataload import dataloadutils


class UFBertExampleLoader:
    def __init__(self, bert_tokenizer, mention_files, label_files, bert_max_seq_len,
                 yield_id=False, loop=True, src_tag='UF', skip_tokenization=False):
        self.bert_tokenizer = bert_tokenizer
        self.mention_files = mention_files
        self.label_files = label_files
        self.file_idx = 0
        self.bert_max_seq_len = bert_max_seq_len
        self.yield_id = yield_id
        self.loop = loop
        self.src_tag = src_tag
        self.skip_tokenization = skip_tokenization

    def __iter__(self):
        return self.next_example()

    def next_example(self):
        is_first_iter = True
        while is_first_iter or self.loop:
            ml_loader = iter(dataloadutils.UFMentionLabelLoader(
                self.mention_files, self.label_files, yield_id=self.yield_id))

            for data in ml_loader:
                mid = None
                if self.yield_id:
                    mid, mention, lobj = data
                else:
                    mention, lobj = data

                mask_idx = None
                # if self.skip_tokenization or bert_tok_id_seq is not None:
                raw_example = {
                    'y_str': mention['y_str'],
                    'mention_span': mention['mention_span'],
                    'left_context_token': mention['left_context_token'],
                    'right_context_token': mention['right_context_token'],
                }

                bert_tok_id_seq, token_type_id_seq = dataloadutils.uf_example_to_qic_bert_token_id_seq(
                    self.bert_tokenizer, raw_example, self.bert_max_seq_len
                )

                sample = {
                    'bert_token_id_seq': bert_tok_id_seq,
                    'token_type_id_seq': token_type_id_seq,
                    'y_str': raw_example['y_str'],
                    'label_obj': lobj,
                    'raw_example': raw_example,
                    'src': self.src_tag
                }
                if mask_idx is not None:
                    sample['mask_idx'] = mask_idx
                if self.yield_id:
                    sample['mid'] = mid

                yield sample

            is_first_iter = False


def spanm_to_uf_mention_example(text, span, labels):
    pbeg, pend = span
    left_cxt_tokens = text[:pbeg].strip().split(' ') if text else []
    right_cxt_tokens = text[pend:].strip().split(' ') if text else []
    return {
        'y_str': labels,
        'mention_span': text[pbeg:pend],
        'left_context_token': left_cxt_tokens,
        'right_context_token': right_cxt_tokens,
    }


def load_bertet_examples(tokenizer, data_file, max_seq_len):
    from utils import datautils

    examples = list()
    for x in datautils.json_obj_iter(data_file):
        tok_type_ids = None
        bert_tok_id_seq, tok_type_ids = dataloadutils.uf_example_to_qic_bert_token_id_seq(
            tokenizer, x, max_seq_len)
        # print(tokenizer.convert_ids_to_tokens(bert_tok_id_seq))
        mask_idx = bert_tok_id_seq.index(tokenizer.mask_token_id)
        examples.append({
            'token_id_seq': bert_tok_id_seq,
            'token_type_ids': tok_type_ids,
            'mask_idx': mask_idx,
            'labels': x['y_str']
        })
        # break
    return examples


class UFBertSpanMExampleLoader:
    def __init__(
            self, bert_tokenizer, mention_file, ex_label_file,
            bert_max_seq_len,
            yield_id=False,
            loop=True,
            skip_tokenization=False
    ):
        self.bert_tokenizer = bert_tokenizer
        self.mention_file = mention_file
        self.ml_loader = dataloadutils.DHLUFETLoader(mention_file, ex_label_file)
        self.bert_max_seq_len = bert_max_seq_len
        self.yield_id = yield_id
        self.loop = loop
        self.skip_tokenization = skip_tokenization

    def __iter__(self):
        return self.next_example()

    def next_example(self):
        ml_iter = iter(self.ml_loader)
        while True:
            mention, lobj = None, None
            try:
                mention, lobj = next(ml_iter)
            except StopIteration:
                if self.loop:
                    ml_iter = iter(self.ml_loader)
                else:
                    break

            if lobj is None:
                continue

            # dataloadutils.spanm_mention_to_sm_bert(self.bert_tokenizer, mention, self.bert_max_seq_len)
            mask_idx = None
            bert_tok_id_seq = None
            mid = mention['id']
            text = mention['text']
            mspan = mention['span']
            raw_example = spanm_to_uf_mention_example(text, mspan, [])
            bert_tok_id_seq, token_type_id_seq = dataloadutils.uf_example_to_qic_bert_token_id_seq(
                self.bert_tokenizer, raw_example, self.bert_max_seq_len)
            sample = {
                'bert_token_id_seq': bert_tok_id_seq,
                'token_type_id_seq': token_type_id_seq,
                'label_obj': lobj,
                'raw_example': raw_example,
                'src': 'PN'
            }
            if mask_idx is not None:
                sample['mask_idx'] = mask_idx
            if self.yield_id:
                sample['mid'] = mid
            yield sample


class MultiSrcDataset(IterableDataset):
    def __init__(self, example_loaders, weights):
        super(MultiSrcDataset).__init__()
        self.example_loaders = example_loaders
        self.weights = weights
        weight_sum = sum(self.weights)
        self.p = [w / weight_sum for w in self.weights]
        self.n_srcs = len(self.example_loaders)

    def __iter__(self):
        return self.next_example()

    def next_example(self):
        example_iters = [iter(loader) for loader in self.example_loaders]
        while True:
            # yield 'foo'
            idx = np.random.choice(self.n_srcs, p=self.p)
            try:
                example = next(example_iters[idx])
            except StopIteration:
                return None
            yield example
