import random
import numpy as np
from torch.utils.data import IterableDataset
from dataload import dataloadutils
from utils import utils


class DirectExampleLoader:
    def __init__(self, examples, loop=True):
        self.examples = examples
        self.loop = loop

    def __iter__(self):
        return self.next_example()

    def next_example(self):
        continue_iter = True
        while continue_iter:
            random.shuffle(self.examples)
            for x in self.examples:
                x['src'] = 'DR'
                yield x
            continue_iter = self.loop


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
        mask_idx = bert_tok_id_seq.index(tokenizer.mask_token_id)
        examples.append({
            'token_id_seq': bert_tok_id_seq,
            'token_type_ids': tok_type_ids,
            'mask_idx': mask_idx,
            'labels': x['y_str']
        })
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

            mask_idx = None
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


class MultiSrcExampleLoader:
    def __init__(self, example_loaders, weights):
        self.example_loaders = example_loaders
        self.weights = weights
        weight_sum = sum(self.weights)
        self.p = [w / weight_sum for w in self.weights]

    def __iter__(self):
        return self.next_example()

    def next_example(self):
        n_srcs = len(self.example_loaders)
        example_iters = [iter(loader) for loader in self.example_loaders]
        while True:
            idx = np.random.choice(n_srcs, p=self.p)
            try:
                example = next(example_iters[idx])
            except StopIteration:
                break

            yield example


def get_fine_types(labels):
    types = set()
    n_labels = len(labels)
    for i in range(n_labels):
        cur_label = labels[i]
        flg = True
        for j in range(n_labels):
            if i == j:
                continue
            if labels[j].startswith(cur_label):
                flg = False
                break
        if flg:
            types.add(cur_label)
    return list(types)


class AliToSpanMExampleLoader:
    def __init__(self, data_file, remove_other=False, single_path=False):
        self.data_file = data_file
        self.remove_other = remove_other
        self.single_path = single_path

    def __iter__(self):
        return self.next_example()

    def next_example(self):
        from utils import datautils
        for i, x in enumerate(datautils.json_obj_iter(self.data_file)):
            lcxt, rcxt = x['left_context_text'], x['right_context_text']
            mstr = x['word']
            labels = x['y_category']
            if self.single_path:
                labels = utils.get_fine_types(labels)
            if self.remove_other and len(labels) > 1 and '/other' in labels:
                labels = [label for label in labels if label != '/other']

            text = lcxt
            pbeg_ch = 0
            if len(text) > 0:
                pbeg_ch = len(text) + 1
                text += ' ' + mstr + ' ' + rcxt
            else:
                text = mstr + ' ' + rcxt
            pend_ch = pbeg_ch + len(mstr)

            yield {'text': text, 'span': (pbeg_ch, pend_ch), 'mstr': mstr, 'labels': labels}
