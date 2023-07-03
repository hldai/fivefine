import logging
from functools import partial
import torch
import numpy as np
from transformers import BertTokenizer
from dataload import exampleload
from dataload.exampleload import load_bertet_examples
from dataload import batchload, dataloadutils
from dataload.batchload import bert_batch_collect, train_batch_list_iter
from models.bertet import TypeTokenBertET
from models import modelutils
from utils import datautils, bertutils
from tasks.uftaskutils import eval_uf


class TrainConfig:
    def __init__(
            self, device, lr=3e-5, w_decay=0.01, batch_size=8, eval_batch_size=32,
            n_steps=20000, max_seq_len=128,
            gacc_step=1,
            n_extra_labels=5,
            pos_prob_thres=0.9,
            neg_prob_thres=0.01,
            weak_to_pos_thres=0.7,
            weak_weight=1.0,
            teacher_model_str=None,
            model_str=None,
            eval_interval=100,
            src_load_weights=(0.4, 0.2, 0.2, 0.2),
    ):
        self.device = device
        self.gacc_step = gacc_step
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.lr = lr
        self.w_decay = w_decay
        self.n_steps = n_steps
        self.pos_prob_thres = pos_prob_thres
        self.neg_prob_thres = neg_prob_thres
        self.weak_to_pos_thres = weak_to_pos_thres
        self.weak_weight = weak_weight
        self.max_seq_len = max_seq_len
        self.n_extra_labels = n_extra_labels
        self.teacher_model_str = teacher_model_str
        self.model_str = model_str
        self.eval_interval = eval_interval
        self.src_load_weights = src_load_weights


class WeakSTExampleLoader:
    def __init__(
            self, device, tokenizer, type_id_dict,
            teacher_example_loader, teacher_model, n_extra_labels,
            pos_prob_thres,
            neg_prob_thres,
            type_vocab=None,
            teacher_batch_size=16,
            max_seq_len=128
    ):
        self.teacher_example_loader = teacher_example_loader
        self.device = device
        self.tokenizer = tokenizer
        self.type_id_dict = type_id_dict
        self.type_vocab = type_vocab
        self.teacher_example_loader = teacher_example_loader
        self.teacher_model = teacher_model
        self.n_extra_labels = n_extra_labels
        self.pos_prob_thres = pos_prob_thres
        self.neg_prob_thres = neg_prob_thres
        self.teacher_batch_size = teacher_batch_size
        self.max_seq_len = max_seq_len
        self.skip_flg = False

    def __iter__(self):
        return self.next_example()

    def batch_example_iter(self):
        batch_examples = list()
        for x in self.teacher_example_loader:
            batch_examples.append(x)
            if len(batch_examples) == self.teacher_batch_size:
                yield batch_examples
                batch_examples = list()
        if len(batch_examples) > 0:
            yield batch_examples

    def next_example(self):
        for batch_examples in self.batch_example_iter():
            if self.skip_flg:
                for _ in batch_examples:
                    if self.skip_flg:
                        yield None
                    else:
                        break
                continue

            batch_size = len(batch_examples)

            token_id_seqs, mask_idxs = list(), list()
            for x in batch_examples:
                bert_token_id_seq = x['bert_token_id_seq']
                mask_idx = bert_token_id_seq.index(self.tokenizer.mask_token_id)
                token_id_seqs.append(bert_token_id_seq)
                mask_idxs.append(mask_idx)

            with torch.no_grad():
                id_seqs_tensor, attn_mask = modelutils.pad_id_seqs(
                    token_id_seqs, self.device, self.tokenizer.pad_token_id)
                logits, _ = self.teacher_model(id_seqs_tensor, attn_mask, mask_idxs)
                probs_batch = torch.sigmoid(logits).data.cpu().numpy()

            pos_labels_list = [list() for _ in range(batch_size)]
            pos_idxs = np.argwhere(probs_batch > self.pos_prob_thres)
            for pi, pj in pos_idxs:
                pos_labels_list[pi].append(pj)
            uncertain_labels_list = [list() for _ in range(batch_size)]
            uncertain_idxs = np.argwhere(
                ((probs_batch > self.neg_prob_thres) & (probs_batch < self.pos_prob_thres)))

            origin_labels_list = [x['raw_example']['y_str'] for x in batch_examples]
            origin_labels_list = [[self.type_id_dict.get(t, -1) for t in labels] for labels in origin_labels_list]
            origin_labels_list = [[tid for tid in labels if tid > -1] for labels in origin_labels_list]
            for pi, pj in uncertain_idxs:
                uncertain_labels_list[pi].append(pj)

            for i, x in enumerate(batch_examples):
                pos_labels, uncertain_labels = pos_labels_list[i], uncertain_labels_list[i]
                lobj = x['label_obj']
                ex_labels = list() if lobj is None else lobj['tids']
                if len(ex_labels) > self.n_extra_labels:
                    ex_labels = ex_labels[:self.n_extra_labels]
                label_probs_dict = dict()
                origin_labels = origin_labels_list[i]
                for tid in set(origin_labels + ex_labels + pos_labels + uncertain_labels):
                    label_probs_dict[tid] = probs_batch[i][tid]
                if len(origin_labels) + len(pos_labels) > 0:
                    yield {
                        'raw_example': x['raw_example'],
                        'bert_token_id_seq': x['bert_token_id_seq'],
                        'origin_labels': origin_labels,
                        'ex_labels': ex_labels,
                        'model_pos_labels': pos_labels,
                        'uncertain_labels': uncertain_labels,
                        'label_probs_dict': label_probs_dict,
                        'src': x['src'],
                    }


def get_pos_and_uncertain_labels(
        original_labels, mlm_labels, model_pos_labels, model_uncertain_labels, tid_prob_dict, weak_to_pos_thres,
        is_from_hw=False):
    pos_labels_set = set(model_pos_labels)

    weak_labels = set()
    if original_labels is not None:
        if is_from_hw:
            if len(original_labels) == 1:
                pos_labels_set.update(original_labels)
            else:
                weak_labels.update(original_labels)
        else:
            weak_labels.update(original_labels)
    if mlm_labels is not None:
        weak_labels.update(mlm_labels)

    # cnt = 0
    for tid in weak_labels:
        if tid_prob_dict[tid] > weak_to_pos_thres:
            pos_labels_set.add(tid)

    uncertain_labels = [tid for tid in model_uncertain_labels if tid not in pos_labels_set]
    return list(pos_labels_set), uncertain_labels


class TypeTokenUFSTExampleLoader:
    def __init__(
            self, tokenizer, type_id_dict, src_example_loaders, src_load_weights,
            weak_to_pos_thres, max_seq_len
    ):
        self.tokenizer = tokenizer
        self.type_id_dict = type_id_dict
        self.src_example_loaders = src_example_loaders
        self.example_loader = exampleload.MultiSrcExampleLoader(src_example_loaders, src_load_weights)
        self.weak_to_pos_thres = weak_to_pos_thres
        self.max_seq_len = max_seq_len

    def set_skip(self):
        for loader in self.src_example_loaders:
            if isinstance(loader, WeakSTExampleLoader):
                loader.skip_flg = True

    def unset_skip(self):
        for loader in self.src_example_loaders:
            if isinstance(loader, WeakSTExampleLoader):
                loader.skip_flg = False

    def __iter__(self):
        return self.next_example()

    def next_example(self):
        for src_example in self.example_loader:
            if src_example is None:
                yield None
                continue

            uncertain_labels = list()
            if src_example['src'] == 'DR':
                bert_token_id_seq, token_type_ids = dataloadutils.uf_example_to_qic_bert_token_id_seq(
                    self.tokenizer, src_example, self.max_seq_len
                )
                pos_labels = [self.type_id_dict[t] for t in src_example['y_str']]
            else:
                bert_token_id_seq = src_example['bert_token_id_seq']
                pos_labels, uncertain_labels = get_pos_and_uncertain_labels(
                    src_example['origin_labels'], src_example['ex_labels'], src_example['model_pos_labels'],
                    src_example['uncertain_labels'], src_example['label_probs_dict'],
                    self.weak_to_pos_thres
                )

            mask_idx = bert_token_id_seq.index(self.tokenizer.mask_token_id)
            yield {
                'bert_token_id_seq': bert_token_id_seq,
                'mask_idx': mask_idx,
                'pos_labels': pos_labels,
                'uncertain_labels': uncertain_labels,
                'src': src_example['src']
            }


class PartialBCELoss(torch.nn.Module):
    def __init__(self, mean_loss=True):
        super(PartialBCELoss, self).__init__()
        self.mean_loss = mean_loss
        self.log_sigmoid = torch.nn.LogSigmoid()

    def forward(self, logits, targets, targets_mask, weights=None):
        pos_vals = -targets * self.log_sigmoid(logits)
        neg_vals = -self.log_sigmoid(-logits) * (1 - targets)
        vals = pos_vals + neg_vals

        norm_val = torch.sum(targets_mask, dim=1) + 1e-5
        losses = torch.sum(vals * targets_mask, dim=-1) / norm_val
        if weights is not None:
            losses *= weights

        if self.mean_loss:
            loss = torch.mean(losses)
        else:
            loss = torch.sum(losses)
        return loss


def get_st_loss_for_batch(
    tc: TrainConfig, loss_obj, loss_obj_weak, model, batch, weak_weight,
    norm_strong, norm_weak
):
    loss_strong = torch.tensor(0.0, dtype=torch.float32, device=tc.device)
    if batch['input_ids_strong'] is not None:
        logits, mlm_logits_strong = model(
            batch['input_ids_strong'], batch['attn_mask_strong'], batch['mask_idxs_strong'])
        target_mask = batch['target_mask_strong']
        labels = batch['pos_labels_strong']
        loss_strong = loss_obj(logits, labels, target_mask)

    loss_weak = torch.tensor(0.0, dtype=torch.float32, device=tc.device)
    if batch['input_ids_weak'] is not None:
        target_mask = batch['target_mask_weak']
        labels = batch['pos_labels_weak']
        logits, mlm_logits_weak = model(batch['input_ids_weak'], batch['attn_mask_weak'], batch['mask_idxs_weak'])

        loss_weak = loss_obj_weak(logits, labels, target_mask)
    if norm_strong > 0:
        loss_strong /= norm_strong
    if norm_weak > 0:
        loss_weak /= norm_weak
    loss = loss_strong + weak_weight * loss_weak

    return loss


class UFSTTrainer:
    def __init__(self, tc: TrainConfig, type_vocab_file, uf_tdt_files, teacher_model_file,
                 st_data_files, load_model_file, save_model_file):
        self.tc = tc

        self.type_vocab, self.type_id_dict = datautils.load_vocab_file(type_vocab_file)
        self.n_types = len(self.type_vocab)

        self.teacher_model_file = teacher_model_file
        self.load_model_file = load_model_file
        self.save_model_file = save_model_file
        self.tdt_files = uf_tdt_files
        self._step = 0
        self._losses = list()
        self.st_data_files = st_data_files

        self.teacher_tokenizer = BertTokenizer.from_pretrained(tc.teacher_model_str)
        self.tokenizer = BertTokenizer.from_pretrained(tc.model_str)
        self.teacher_model = None
        self.model = None
        self.optimizer = None
        self.reload_model(load_model_file, teacher_model_file)
        self.skipping = False

    def reload_model(self, load_model_file, teacher_model_file):
        tc = self.tc
        self.teacher_model = TypeTokenBertET.from_trained(teacher_model_file, tc.teacher_model_str)
        self.teacher_model.init_type_hiddens(self.teacher_tokenizer, self.type_vocab, device=tc.device)
        self.teacher_model.to(self.tc.device)
        self.teacher_model.eval()
        logging.info('load teacher model from {}'.format(teacher_model_file))

        if load_model_file is None:
            self.model = TypeTokenBertET(tc.model_str, tc.model_str)
        else:
            logging.info('load_model_file={}'.format(load_model_file))
            self.model = TypeTokenBertET.from_trained(load_model_file, tc.model_str)
        # self.model = TypeTokenBertUF(self.n_types, tc.model_str, bert_dir=tc.model_str)
        self.model.to(self.tc.device)
        self.model.init_type_hiddens(self.tokenizer, self.type_vocab)
        self.model.train()

        if self.optimizer is not None:
            self.optimizer = bertutils.get_bert_adam_optim(
                list(self.model.named_parameters()), learning_rate=tc.lr, w_decay=tc.w_decay)
            self.optimizer.zero_grad()

    def batch_collect_fn(self, examples):
        if self.skipping:
            examples = [x for x in examples if x is not None]
        batch_strong = [x for x in examples if x['src'] == 'DR']
        batch_weak = [x for x in examples if x['src'] != 'DR']

        def get_model_input(batch_examples):
            if len(batch_examples) == 0:
                return None, None, None, None
            token_id_seqs = [x['bert_token_id_seq'] for x in batch_examples]
            input_ids, attn_mask = modelutils.pad_id_seqs(token_id_seqs, self.tc.device, self.tokenizer.pad_token_id)
            mask_idxs = [x['mask_idx'] for x in batch_examples]
            # return input_ids, attn_mask, mask_idxs, mlm_labels
            return input_ids, attn_mask, mask_idxs

        def get_labels_tensor(batch_examples):
            if len(batch_examples) == 0:
                return None
            labels = modelutils.onehot_encode_batch([x['pos_labels'] for x in batch_examples], self.n_types)
            return torch.tensor(labels, dtype=torch.float32, device=self.tc.device)

        input_ids_strong, attn_mask_strong, mask_idxs_strong = get_model_input(batch_strong)
        input_ids_weak, attn_mask_weak, mask_idxs_weak = get_model_input(batch_weak)

        target_mask_strong, target_mask_weak = None, None
        if len(batch_strong) > 0:
            target_mask_strong = np.ones((len(batch_strong), self.n_types), dtype=np.float32)
            target_mask_strong = torch.tensor(target_mask_strong, dtype=torch.float32, device=self.tc.device)
        if len(batch_weak) > 0:
            target_mask_weak = np.ones((len(batch_weak), self.n_types), dtype=np.float32)
            for i, x in enumerate(batch_weak):
                target_mask_weak[i][x['uncertain_labels']] = 0
            target_mask_weak = torch.tensor(target_mask_weak, dtype=torch.float32, device=self.tc.device)

        pos_labels_strong = get_labels_tensor(batch_strong)
        pos_labels_weak = get_labels_tensor(batch_weak)

        return {
            'n_strong': len(batch_strong),
            'n_weak': len(batch_weak),
            'input_ids_strong': input_ids_strong,
            'attn_mask_strong': attn_mask_strong,
            'mask_idxs_strong': mask_idxs_strong,
            'pos_labels_strong': pos_labels_strong,
            'target_mask_strong': target_mask_strong,
            'input_ids_weak': input_ids_weak,
            'attn_mask_weak': attn_mask_weak,
            'mask_idxs_weak': mask_idxs_weak,
            'pos_labels_weak': pos_labels_weak,
            'target_mask_weak': target_mask_weak,
        }

    def train(self):
        tc = self.tc
        src_load_weights = tc.src_load_weights
        logging.info(' '.join(['{}={}'.format(k, v) for k, v in vars(tc).items()]))
        logging.info('src_load_weights: {}'.format(' '.join([str(v) for v in src_load_weights])))

        dev_examples = load_bertet_examples(self.tokenizer, self.tdt_files['dev'], tc.max_seq_len)
        batch_collect_fn = partial(bert_batch_collect, tc.device, self.type_id_dict, self.tokenizer.pad_token_id)
        dev_batch_loader = batchload.IterExampleBatchLoader(
            dev_examples, tc.eval_batch_size, n_iter=1, collect_fn=batch_collect_fn)

        batch_collect_fn = partial(bert_batch_collect, tc.device, self.type_id_dict, self.tokenizer.pad_token_id)

        test_examples = load_bertet_examples(self.tokenizer, self.tdt_files['test'], tc.max_seq_len)
        test_batch_loader = batchload.IterExampleBatchLoader(
            test_examples, tc.eval_batch_size, n_iter=1, collect_fn=batch_collect_fn)

        train_examples = datautils.read_json_objs(self.tdt_files['train'])
        strong_example_loader = exampleload.DirectExampleLoader(train_examples, loop=True)

        el_teacher_example_loader = exampleload.UFBertExampleLoader(
            self.teacher_tokenizer, [self.st_data_files['el_train_file']],
            [self.st_data_files['el_extra_label_file']], self.tc.max_seq_len, src_tag='EL')
        open_teacher_example_loader = exampleload.UFBertExampleLoader(
            self.teacher_tokenizer, self.st_data_files['open_train_files'],
            self.st_data_files['open_extra_label_files'], self.tc.max_seq_len, src_tag='OPEN')
        pn_teacher_example_loader = exampleload.UFBertSpanMExampleLoader(
            self.teacher_tokenizer, self.st_data_files['pronoun_mention_file'],
            self.st_data_files['pronoun_label_file'], self.tc.max_seq_len
        )

        src_example_loaders = [strong_example_loader]
        for teacher_example_loader in [el_teacher_example_loader, open_teacher_example_loader,
                                       pn_teacher_example_loader]:
            src_example_loaders.append(WeakSTExampleLoader(
                tc.device, self.tokenizer, self.type_id_dict, teacher_example_loader, self.teacher_model,
                tc.n_extra_labels, tc.pos_prob_thres, tc.neg_prob_thres, type_vocab=self.type_vocab
            ))

        st_example_loader = TypeTokenUFSTExampleLoader(
            self.tokenizer, self.type_id_dict, src_example_loaders, src_load_weights, tc.weak_to_pos_thres,
            tc.max_seq_len
        )

        train_batch_loader = batchload.IterExampleBatchLoader(
            st_example_loader, tc.batch_size,
            n_steps=tc.n_steps * tc.gacc_step, collect_fn=self.batch_collect_fn)

        loss_obj_strong = PartialBCELoss(mean_loss=False)
        loss_obj_weak = PartialBCELoss(mean_loss=False)
        mlm_loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token

        self.optimizer = bertutils.get_bert_adam_optim(
            list(self.model.named_parameters()), learning_rate=tc.lr, w_decay=tc.w_decay)
        self.optimizer.zero_grad()

        step = 0
        last_best_step = 0
        losses = list()
        self.model.eval()
        best_f1 = -1
        self_train_it = 0
        last_save_model_file = None
        for batch_list in train_batch_list_iter(train_batch_loader, tc.gacc_step):
            n_strong = sum([batch['n_strong'] for batch in batch_list])
            n_weak = sum([batch['n_weak'] for batch in batch_list])
            for batch in batch_list:
                loss = get_st_loss_for_batch(
                    tc, loss_obj_strong, loss_obj_weak, self.model, batch,
                    weak_weight=tc.weak_weight, norm_strong=n_strong, norm_weak=n_weak)
                losses.append(loss.data.cpu().numpy())
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            step += 1
            if step % tc.eval_interval == 0:
                loss_val = sum(losses)
                losses = list()
                self.model.eval()
                p, r, f1, _ = eval_uf(self.model, self.type_vocab, dev_batch_loader)
                # print(step, loss_val, p, r, f1)
                best_tag = '*' if f1 > best_f1 else ''
                logging.info('i={} l={:.4f} p={:.4f} r={:.4f} f1={:.4f}{}'.format(step, loss_val, p, r, f1, best_tag))
                if f1 > best_f1:
                    pt, rt, f1t, _ = eval_uf(self.model, self.type_vocab, test_batch_loader)
                    logging.info('TEST p={:.4f} r={:.4f} f1={:.4f}'.format(pt, rt, f1t))

                if f1 > best_f1 and self.save_model_file is not None:
                    last_save_model_file = self.save_model_file.format(self_train_it)
                    modelutils.save_model(self.model, last_save_model_file, False)
                if f1 > best_f1:
                    last_best_step = step
                    best_f1 = f1

                self.model.train()

                if step > 5000 and step - last_best_step > 5000:
                    self.reload_model(self.load_model_file, last_save_model_file)
                    for loader in src_example_loaders:
                        if isinstance(loader, WeakSTExampleLoader):
                            loader.teacher_model = self.teacher_model
                    step = 0
                    last_best_step = 0
                    self_train_it += 1
                    best_f1 = -1
                    if self_train_it == 3:
                        break
