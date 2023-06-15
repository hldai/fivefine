import json


class DHLUFETLoader:
    def __init__(self, mention_file, label_file):
        self.mention_file = mention_file
        self.label_file = label_file

    def __iter__(self):
        return self.mention_label_gen()

    def mention_label_gen(self):
        fm = open(self.mention_file, encoding='utf-8')
        fl = open(self.label_file, encoding='utf-8')
        cur_label_obj = None
        for i, line_m in enumerate(fm):
            mention = json.loads(line_m)
            mention_id = mention['id']
            if (cur_label_obj is None or cur_label_obj['id'] < mention_id) and fl is not None:
                try:
                    line_l = next(fl)
                    cur_label_obj = json.loads(line_l)
                except StopIteration:
                    fl.close()
                    fl = None
                    break
            if cur_label_obj['id'] == mention_id:
                yield mention, cur_label_obj
            else:
                if cur_label_obj['id'] < mention_id:
                    print(i, mention)
                    print(cur_label_obj)
                    print()
                assert cur_label_obj['id'] > mention_id
                yield mention, None
        fm.close()
        if fl is not None:
            fl.close()


class UFMentionLabelLoader:
    def __init__(self, mention_files, extra_label_files, yield_id=False):
        self.mention_files = mention_files
        self.extra_label_files = extra_label_files
        self.yield_id = yield_id

    def __iter__(self):
        return self.mention_label_gen()

    def mention_label_gen(self):
        for mention_file, extra_label_file in zip(self.mention_files, self.extra_label_files):
            fm = open(mention_file, encoding='utf-8')
            fl = open(extra_label_file, encoding='utf-8') if extra_label_file is not None else None
            cur_label_obj = None
            for i, line_m in enumerate(fm):
                mention = json.loads(line_m)
                if extra_label_file is None:
                    if self.yield_id:
                        yield i, mention, None
                    else:
                        yield mention, None
                else:
                    if (cur_label_obj is None or cur_label_obj['id'] < i) and fl is not None:
                        try:
                            line_l = next(fl)
                            cur_label_obj = json.loads(line_l)
                        except StopIteration:
                            fl.close()
                            fl = None
                    r_label_obj = None
                    if cur_label_obj['id'] == i:
                        r_label_obj = cur_label_obj
                    else:
                        assert cur_label_obj['id'] > i
                    if self.yield_id:
                        yield i, mention, r_label_obj
                    else:
                        yield mention, r_label_obj
            fm.close()
            if fl is not None:
                fl.close()


def uf_example_to_qic_bert_token_id_seq(tokenizer, example, max_seq_len, gen_neighbor=False):
    mstr_prefix_token_seq = tokenizer.tokenize('[')
    mstr_suffix_token_seq = tokenizer.tokenize('] (Type: [MASK])')
    mstr_query_extra_len = len(mstr_prefix_token_seq) + len(mstr_suffix_token_seq)

    left_nw_suffix_token_seq, right_nw_suffix_token_seq = [], []
    if gen_neighbor:
        left_nw_suffix_token_seq = tokenizer.tokenize('] (Left: [MASK])')
        right_nw_suffix_token_seq = tokenizer.tokenize('] (Right: [MASK])')
        assert len(left_nw_suffix_token_seq) <= len(mstr_suffix_token_seq)
        assert len(right_nw_suffix_token_seq) <= len(mstr_suffix_token_seq)

    lcxt, rcxt = ' '.join(example['left_context_token']), ' '.join(example['right_context_token'])
    mstr = example['mention_span']

    lcxt_token_seq, rcxt_token_seq = tokenizer.tokenize(lcxt), tokenizer.tokenize(rcxt)

    mstr_token_seq = tokenizer.tokenize(mstr)
    mstr_query_token_seq = mstr_prefix_token_seq + mstr_token_seq + mstr_suffix_token_seq
    mstr_query_token_type_ids = [0] * len(mstr_prefix_token_seq) + [1] * len(mstr_token_seq) + [0] * len(
        mstr_suffix_token_seq)

    if len(lcxt_token_seq) + len(rcxt_token_seq) + len(mstr_query_token_seq) + 2 > max_seq_len:
        rcxt_len = max(1, max_seq_len - len(lcxt_token_seq) - len(mstr_query_token_seq) - 2)
        rcxt_token_seq = rcxt_token_seq[:rcxt_len]

        if len(lcxt_token_seq) + len(rcxt_token_seq) + len(mstr_query_token_seq) + 2 > max_seq_len:
            lcxt_len = max(1, max_seq_len - len(rcxt_token_seq) - len(mstr_query_token_seq) - 2)
            cur_len = len(lcxt_token_seq)
            lcxt_token_seq = lcxt_token_seq[cur_len - lcxt_len:]

        if len(lcxt_token_seq) + len(rcxt_token_seq) + len(mstr_query_token_seq) + 2 > max_seq_len:
            lcxt_token_seq = []
            rcxt_token_seq = []
            mstr_seq_len = min(
                max_seq_len - 2 - mstr_query_extra_len, len(mstr_query_token_seq) - mstr_query_extra_len)
            mstr_token_seq = mstr_token_seq[:mstr_seq_len]
            mstr_query_token_seq = mstr_prefix_token_seq + mstr_token_seq + mstr_suffix_token_seq
            mstr_query_token_type_ids = [0] * len(mstr_prefix_token_seq) + [1] * len(mstr_token_seq) + [0] * len(
                mstr_suffix_token_seq)

    full_token_seq = lcxt_token_seq + mstr_query_token_seq + rcxt_token_seq
    full_token_type_ids = [0] * len(lcxt_token_seq) + mstr_query_token_type_ids + [0] * len(rcxt_token_seq)
    full_token_seq = [tokenizer.cls_token] + full_token_seq + [tokenizer.sep_token]
    full_token_type_ids = [0] + full_token_type_ids + [0]

    if gen_neighbor:
        left_token_id = tokenizer.cls_token_id
        right_token_id = tokenizer.sep_token_id
        if len(lcxt_token_seq) > 0:
            left_token_id = tokenizer.convert_tokens_to_ids([lcxt_token_seq[-1]])[0]
        if len(rcxt_token_seq) > 0:
            right_token_id = tokenizer.convert_tokens_to_ids([rcxt_token_seq[0]])[0]
        ln_token_seq = [tokenizer.cls_token] + lcxt_token_seq + (
                mstr_prefix_token_seq + mstr_token_seq + left_nw_suffix_token_seq) + (
                rcxt_token_seq + [tokenizer.sep_token])
        rn_token_seq = [tokenizer.cls_token] + lcxt_token_seq + (
                mstr_prefix_token_seq + mstr_token_seq + right_nw_suffix_token_seq) + (
                rcxt_token_seq + [tokenizer.sep_token])

        full_token_id_seq = tokenizer.convert_tokens_to_ids(full_token_seq)
        ln_token_id_seq = tokenizer.convert_tokens_to_ids(ln_token_seq)
        rn_token_id_seq = tokenizer.convert_tokens_to_ids(rn_token_seq)
        return full_token_id_seq, ln_token_id_seq, rn_token_id_seq, left_token_id, right_token_id, full_token_type_ids

    return tokenizer.convert_tokens_to_ids(full_token_seq), full_token_type_ids
