import logging
import os
import random
from functools import partial
import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertTokenizer
from torch.nn import CrossEntropyLoss
from utils import datautils, bertutils
from models.bertet import TypeTokenBertET
from models import modelutils
from dataload import exampleload, dataloadutils, batchload
from dataload.batchload import train_batch_list_iter
from dataload.exampleload import load_bertet_examples


class TrainConfig:
    def __init__(
            self, device, lr=3e-5, w_decay=0.01, batch_size=8,
            gacc_step=1,
            eval_batch_size=32,
            n_steps=20000, max_seq_len=128, eval_interval=100,
            mlm_lamb=0.1, nw_lamb=0.1, n_ex_types=10,
            weight_for_origin_label=5.0,
            neighbor_words=True,
            nw_train_interval=2,
            token_type=False,
            save_interval=10000):
        self.device = device
        self.batch_size = batch_size
        self.gacc_step = gacc_step
        self.eval_batch_size = eval_batch_size
        self.lr = lr
        self.w_decay = w_decay
        self.n_steps = n_steps
        self.max_seq_len = max_seq_len
        self.weight_for_origin_label = weight_for_origin_label
        self.mlm_lamb = mlm_lamb
        self.nw_lamb = nw_lamb
        self.nw_train_interval = nw_train_interval
        self.n_ex_types = n_ex_types
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.neighbor_words = neighbor_words
        self.token_type = token_type


class WeightedBCELoss(torch.nn.Module):
    def __init__(self, neg_scale=-1, bce_sum=False):
        super(WeightedBCELoss, self).__init__()
        self.log_sigmoid = torch.nn.LogSigmoid()
        self.neg_scale = neg_scale
        self.bce_sum = bce_sum

    def forward(self, logits, targets, target_weights):
        neg_vals = self.log_sigmoid(-logits) * (1 - targets)
        if self.neg_scale > 0:
            neg_vals *= self.neg_scale
        vals = -targets * self.log_sigmoid(logits) - neg_vals
        if self.bce_sum:
            losses = torch.sum(vals * target_weights, dim=-1)
        else:
            losses = torch.sum(vals * target_weights, dim=-1) / logits.size()[1]
        return torch.mean(losses)


class DataLoaderCollateFn:
    def __init__(self, tc: TrainConfig, tokenizer, type_id_dict, ckpt_step, gacc_step):
        self.tc = tc
        self.device = torch.device('cpu')
        self.tokenizer = tokenizer
        # self.tokenizer = BertTokenizer.from_pretrained(config.BERT_LARGE_MODEL_PATH)
        self.type_id_dict = type_id_dict
        self._step = 0
        self._ckpt_step = ckpt_step
        self.gacc_step = gacc_step
        self.mlm_probability = 0.1
        self.rand_word_rate = 0.1
        self.mlm_type_label_weight = 1.0

    def add_mlm_to_input(self, input_ids):
        return bertutils.add_mlm_to_bert_input(
            input_ids, self.tokenizer, self.device, self.mlm_probability, self.rand_word_rate)

    def collate_fn(self, examples):
        tc = self.tc
        device = self.device
        self._step += 1
        if self._step <= self._ckpt_step * self.gacc_step:
            return {}
        pad_token_id = self.tokenizer.pad_token_id
        # print(examples[0])
        input_ids, mask_idxs = list(), list()
        left_nw_input_ids, left_nw_mask_idxs = list(), list()
        right_nw_input_ids, right_nw_mask_idxs = list(), list()
        left_token_ids, right_token_ids = list(), list()
        tok_type_ids_list = list()
        for x in examples:
            ln_token_id_seq, rn_token_id_seq, left_token_id, right_token_id = None, None, None, None
            if tc.neighbor_words:
                (bert_tok_id_seq, ln_token_id_seq, rn_token_id_seq, left_token_id, right_token_id, tok_type_ids
                 ) = dataloadutils.uf_example_to_qic_bert_token_id_seq(
                    self.tokenizer, x['raw_example'], self.tc.max_seq_len, gen_neighbor=tc.neighbor_words)
            else:
                bert_tok_id_seq, tok_type_ids = dataloadutils.uf_example_to_qic_bert_token_id_seq(
                    self.tokenizer, x['raw_example'], self.tc.max_seq_len, gen_neighbor=tc.neighbor_words)

            try:
                mask_idx = bert_tok_id_seq.index(self.tokenizer.mask_token_id)
            except ValueError:
                if len(bert_tok_id_seq) == 0:
                    bert_tok_id_seq = [
                        self.tokenizer.cls_token_id, self.tokenizer.mask_token_id, self.tokenizer.sep_token_id]
                else:
                    bert_tok_id_seq.insert(-1, self.tokenizer.mask_token_id)
                mask_idx = len(bert_tok_id_seq) - 1
            input_ids.append(bert_tok_id_seq)
            mask_idxs.append(mask_idx)
            if tok_type_ids is not None:
                tok_type_ids_list.append(tok_type_ids)

            if tc.neighbor_words:
                left_nw_input_ids.append(ln_token_id_seq)
                right_nw_input_ids.append(rn_token_id_seq)
                left_nw_mask_idxs.append(ln_token_id_seq.index(self.tokenizer.mask_token_id))
                right_nw_mask_idxs.append(rn_token_id_seq.index(self.tokenizer.mask_token_id))
                left_token_ids.append(left_token_id)
                right_token_ids.append(right_token_id)

        input_ids, attn_mask = modelutils.pad_id_seqs(input_ids, device, pad_token_id)
        token_type_ids = None
        if len(tok_type_ids_list) > 0:
            token_type_ids = modelutils.pad_seq_to_len(tok_type_ids_list, input_ids.size()[1], device)

        mlm_labels = None
        if self.tc.mlm_lamb > 0:
            input_ids, mlm_labels = self.add_mlm_to_input(input_ids)

        labels_list = [x.get('y_str', list()) for x in examples]
        type_ids_list, label_weights_list = list(), list()
        for labels in labels_list:
            type_ids = [self.type_id_dict.get(t, None) for t in labels]
            # type_ids = [0, 1]
            type_ids = [tid for tid in type_ids if tid is not None]
            type_ids_list.append(type_ids)
            label_weights_list.append([self.tc.weight_for_origin_label for _ in type_ids])
        # type_ids_list = [[self.type_id_dict[t] for t in labels] for labels in labels_list]

        for i in range(len(examples)):
            type_ids = type_ids_list[i]
            # print(examples[i])
            lobj = examples[i]['label_obj']
            ex_type_ids = list()
            if lobj is not None:
                ex_type_ids = lobj['tids']
                ex_type_ids = ex_type_ids[:self.tc.n_ex_types]
            for tid in ex_type_ids:
                if tid not in type_ids:
                    type_ids.append(tid)
                    label_weights_list[i].append(self.mlm_type_label_weight)

        batch = {
            'input_ids': input_ids,
            'attn_mask': attn_mask,
            'token_type_ids': token_type_ids,
            'mask_idxs': mask_idxs,
            'mlm_labels': mlm_labels,
            'type_ids_list': type_ids_list,
            'label_weights_list': label_weights_list,
            'labels_list': labels_list,
            'srcs': [x['src'] for x in examples]
        }

        if tc.neighbor_words:
            # print('a', [len(v) for v in left_nw_input_ids])
            # print('b', token_type_ids.size())
            batch['ln_input_ids'] = left_nw_input_ids
            batch['left_nw_mask_idxs'] = left_nw_mask_idxs
            batch['left_token_ids'] = left_token_ids
            batch['rn_input_ids'] = right_nw_input_ids
            batch['right_nw_mask_idxs'] = right_nw_mask_idxs
            batch['right_token_ids'] = right_token_ids

        return batch


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


class WeakUFModelTrainer:
    def __init__(
            self, tc: TrainConfig, type_vocab_file, bert_model_str, uf_data_files,
            load_model_file=None,
            save_model_file_prefix=None):
        device = tc.device
        self.type_vocab, self.type_id_dict = datautils.load_vocab_file(type_vocab_file)
        self.n_types = len(self.type_vocab)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_str)

        self.tc = tc
        self.model = TypeTokenBertET(bert_model_str, bert_model_str)
        self.model.to(device)
        self.model.init_type_hiddens(self.tokenizer, self.type_vocab)

        self.load_model_file = load_model_file
        self.load_ckpt_file = None
        if load_model_file is not None:
            self.load_ckpt_file = os.path.splitext(self.load_model_file)[0] + '.ckpt'
            logging.info('ckpt_file={}'.format(self.load_ckpt_file))
            self.model.load_state_dict(torch.load(load_model_file, map_location='cpu'))

        self.uf_data_files = uf_data_files
        self.save_model_file_prefix = save_model_file_prefix
        # self.mlm_probability = 0.15
        self.mlm_probability = 0.1
        self.rand_word_rate = 0.1
        self.mlm_type_label_weight = 1.0
        self._step = 0
        self._ckpt_step = 0

        self.mlm_loss_fct = CrossEntropyLoss()  # -100 index = padding token

    def _nw_train_step(self, optimizer, batch_list):
        all_examples = list()
        batch_size = 0
        for batch in batch_list:
            batch_size = len(batch['ln_input_ids'])
            ln_input_ids, left_nw_mask_idxs, left_token_ids = (
                batch['ln_input_ids'], batch['left_nw_mask_idxs'], batch['left_token_ids'])
            rn_input_ids, right_nw_mask_idxs, right_token_ids = (
                batch['rn_input_ids'], batch['right_nw_mask_idxs'], batch['right_token_ids'])
            token_type_ids = batch['token_type_ids']
            for i in range(batch_size):
                all_examples.append((ln_input_ids[i], left_nw_mask_idxs[i], left_token_ids[i], token_type_ids[i]))
                all_examples.append((rn_input_ids[i], right_nw_mask_idxs[i], right_token_ids[i], token_type_ids[i]))

        random.shuffle(all_examples)
        all_examples = all_examples[:batch_size]
        input_ids = [x[0] for x in all_examples]
        mask_idxs = [x[1] for x in all_examples]
        target_token_ids = [x[2] for x in all_examples]
        input_ids, attn_mask = modelutils.pad_id_seqs(input_ids, self.tc.device, self.tokenizer.pad_token_id)
        token_type_ids = None
        if self.tc.token_type:
            token_type_ids = [x[3] for x in all_examples]
            token_type_ids = modelutils.pad_seq_to_len(token_type_ids, input_ids.size()[1], self.tc.device)

        nw_logits = self.model(input_ids, attn_mask, mask_idxs, token_type_ids=token_type_ids, mode='nw')

        target_token_ids = torch.tensor(target_token_ids, dtype=torch.long, device=self.tc.device)
        loss = self.tc.nw_lamb * self.mlm_loss_fct(nw_logits, target_token_ids.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.data.cpu().numpy()

    def train(self):
        logging.info(' '.join(['{}={}'.format(k, v) for k, v in vars(self.tc).items()]))
        tc = self.tc
        device = self.tc.device
        if tc.weight_for_origin_label > 0:
            loss_obj = WeightedBCELoss()
        else:
            loss_obj = torch.nn.BCEWithLogitsLoss()
        # loss_obj = torch.nn.BCEWithLogitsLoss()
        optimizer = bertutils.get_bert_adam_optim(
            list(self.model.named_parameters()), learning_rate=tc.lr, w_decay=tc.w_decay)

        el_example_loader = exampleload.UFBertExampleLoader(
            self.tokenizer, [self.uf_data_files['el_train_file']],
            [self.uf_data_files['el_extra_label_file']], tc.max_seq_len, skip_tokenization=True)
        open_example_loader = exampleload.UFBertExampleLoader(
            self.tokenizer, self.uf_data_files['open_train_files'],
            self.uf_data_files['open_extra_label_files'], tc.max_seq_len, skip_tokenization=True)
        pn_example_loader = exampleload.UFBertSpanMExampleLoader(
            self.tokenizer, self.uf_data_files['pronoun_mention_file'],
            self.uf_data_files['pronoun_label_file'], tc.max_seq_len, skip_tokenization=True)

        train_example_loader = exampleload.MultiSrcDataset(
            [el_example_loader, open_example_loader, pn_example_loader], [0.4, 0.3, 0.3])

        dev_examples = load_bertet_examples(self.tokenizer, self.uf_data_files['dev'], tc.max_seq_len)

        ckpt_step = -1
        if self.load_ckpt_file is not None:
            if os.path.exists(self.load_ckpt_file):
                ckpt_data = torch.load(self.load_ckpt_file, map_location='cpu')
                ckpt_step = ckpt_data['step']
                optimizer.load_state_dict(ckpt_data['optim'])
                torch.set_rng_state(ckpt_data['rng_state'])
                torch.cuda.set_rng_state(ckpt_data['cuda_rng_state'], tc.device)
                logging.info('{} loaded. Will start from {}'.format(self.load_ckpt_file, ckpt_step))
            else:
                logging.info('{} not exist'.format(self.load_ckpt_file))

        optimizer.zero_grad()
        self.model.train()
        batch_collect_fn = partial(bert_batch_collect, device, self.type_id_dict, self.tokenizer.pad_token_id)

        train_colate_fn_obj = DataLoaderCollateFn(
            self.tc, self.tokenizer, self.type_id_dict, ckpt_step, tc.gacc_step)
        train_batch_loader = DataLoader(
            train_example_loader, batch_size=tc.batch_size,
            collate_fn=train_colate_fn_obj.collate_fn, num_workers=1,
            prefetch_factor=8, pin_memory=True)

        train_batch_list_it = iter(train_batch_list_iter(train_batch_loader, tc.gacc_step))
        if ckpt_step > 0:
            for i in range(ckpt_step):
                next(train_batch_list_it)

        dev_batch_loader = batchload.IterExampleBatchLoader(
            dev_examples, tc.eval_batch_size, n_iter=1, collect_fn=batch_collect_fn)
        step = max(ckpt_step, 0)
        losses = list()
        nw_losses = list()

        logging.info('step={}'.format(step))

        for batch_list in train_batch_list_it:
            if self.tc.neighbor_words and step % self.tc.nw_train_interval == 0:
                nw_loss = self._nw_train_step(optimizer, batch_list)
                nw_losses.append(nw_loss)
            for batch in batch_list:
                input_ids = batch['input_ids'].to(device)
                attn_mask = batch['attn_mask'].to(device)
                token_type_ids = None
                if tc.token_type:
                    token_type_ids = batch['token_type_ids'].to(device)
                type_ids_list = batch['type_ids_list']
                mask_idxs = batch['mask_idxs']

                cur_batch_size = len(type_ids_list)

                logits, mlm_logits = self.model(input_ids, attn_mask, mask_idxs, token_type_ids=token_type_ids)

                if tc.weight_for_origin_label > 0:
                    label_weights_list = batch['label_weights_list']
                    labels_tensor = torch.zeros((cur_batch_size, self.n_types), dtype=torch.float32, device=tc.device)
                    weights_tensor = torch.ones((cur_batch_size, self.n_types), dtype=torch.float32, device=tc.device)
                    for i, (labels, weights) in enumerate(zip(type_ids_list, label_weights_list)):
                        # print([self.type_vocab[tid] for tid in labels])
                        # print(weights)
                        for tid, weight in zip(labels, weights):
                            labels_tensor[i][tid] = 1
                            weights_tensor[i][tid] = weight
                    loss = loss_obj(logits, labels_tensor, weights_tensor)
                else:
                    labels = modelutils.onehot_encode_batch(type_ids_list, self.n_types)
                    labels = torch.tensor(labels, dtype=torch.float32, device=device)
                    loss = loss_obj(logits, labels)

                if self.tc.mlm_lamb > 0:
                    mlm_labels = batch['mlm_labels'].to(device)
                    masked_lm_loss = self.mlm_loss_fct(
                        mlm_logits.view(-1, self.tokenizer.vocab_size), mlm_labels.view(-1))
                    loss += self.tc.mlm_lamb * masked_lm_loss
                loss /= tc.gacc_step
                # print(loss)
                losses.append(loss.data.cpu().numpy())
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            step += 1
            if step % self.tc.eval_interval == 0:
                loss_val = sum(losses)
                losses = list()
                self.model.eval()
                p, r, f1, _ = eval_uf(self.model, self.type_vocab, dev_batch_loader, token_type=tc.token_type)
                if self.tc.neighbor_words:
                    nw_loss_val = sum(nw_losses)
                    nw_losses = list()
                    logging.info('i={} l={:.4f} nwl={:.4f} p={:.4f} r={:.4f} f1={:.4f}'.format(
                        step, loss_val, nw_loss_val, p, r, f1))
                else:
                    logging.info('i={} l={:.4f} p={:.4f} r={:.4f} f1={:.4f}'.format(step, loss_val, p, r, f1))
                self.model.train()

            if step % self.tc.save_interval == 0 and self.save_model_file_prefix is not None:
                file_name = f'{self.save_model_file_prefix}-{step}.pth'
                torch.save(self.model.state_dict(), file_name)
                logging.info('model saved to {}'.format(file_name))
                # ckpt_name = f'{self.save_model_file_prefix}-{step}.ckpt'
                # ckpt_data = {
                #     'step': step,
                #     'optim': optimizer.state_dict(),
                #     'rng_state': torch.get_rng_state(),
                #     'cuda_rng_state': torch.cuda.get_rng_state()
                # }
                # torch.save(ckpt_data, ckpt_name)


class TypeTokenModelTrainer:
    def __init__(
            self, tc: TrainConfig, bert_model_str, type_vocab_file, data_files,
            load_model_file, save_model_file=None):
        device = tc.device
        self.type_vocab, self.type_id_dict = datautils.load_vocab_file(type_vocab_file)
        self.n_types = len(self.type_vocab)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_str)

        self.data_files = data_files

        self.tc = tc
        if load_model_file is None:
            self.model = TypeTokenBertET(bert_model_str, bert_model_str)
        else:
            logging.info('load_model_file={}'.format(load_model_file))
            self.model = TypeTokenBertET.from_trained(load_model_file, bert_model_str)
        self.model.to(device)
        self.model.init_type_hiddens(self.tokenizer, self.type_vocab)

        self.save_model_file = save_model_file

    def train(self):
        logging.info(' '.join(['{}={}'.format(k, v) for k, v in vars(self.tc).items()]))
        tc = self.tc
        device = self.tc.device

        train_examples = load_bertet_examples(
            self.tokenizer, self.data_files['train'], tc.max_seq_len)
        dev_examples = load_bertet_examples(self.tokenizer, self.data_files['dev'], tc.max_seq_len)
        test_examples = load_bertet_examples(self.tokenizer, self.data_files['test'], tc.max_seq_len)
        random.shuffle(train_examples)

        batch_collect_fn = partial(bert_batch_collect, device, self.type_id_dict, self.tokenizer.pad_token_id)
        train_batch_loader = batchload.IterExampleBatchLoader(
            train_examples, tc.batch_size, n_steps=tc.n_steps, collect_fn=batch_collect_fn)
        dev_batch_loader = batchload.IterExampleBatchLoader(
            dev_examples, tc.eval_batch_size, n_iter=1, collect_fn=batch_collect_fn)
        test_batch_loader = batchload.IterExampleBatchLoader(
            test_examples, tc.eval_batch_size, n_iter=1, collect_fn=batch_collect_fn)

        loss_obj = torch.nn.BCEWithLogitsLoss()
        optimizer = bertutils.get_bert_adam_optim(
            list(self.model.named_parameters()), learning_rate=tc.lr, w_decay=tc.w_decay)

        step = 0
        best_dev_f1 = -1
        losses = list()
        for batch in train_batch_loader:
            input_ids = batch['input_ids']
            attn_mask = batch['attn_mask']
            type_ids_list = batch['type_ids_list']
            mask_idxs = batch['mask_idxs']
            logits, _ = self.model(input_ids, attn_mask, mask_idxs)

            labels = modelutils.onehot_encode_batch(type_ids_list, self.n_types)
            labels = torch.tensor(labels, dtype=torch.float32, device=device)
            loss = loss_obj(logits, labels)
            losses.append(loss.data.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % tc.eval_interval == 0:
                loss_val = sum(losses)
                losses = list()
                self.model.eval()
                p, r, f1, _ = eval_uf(self.model, self.type_vocab, dev_batch_loader)
                best_tag = ''
                if f1 > best_dev_f1:
                    best_tag = '*'
                logging.info('i={} l={:.4f} p={:.4f} r={:.4f} f1={:.4f}{}'.format(
                    step, loss_val, p, r, f1, best_tag))
                if f1 > best_dev_f1:
                    pt, rt, f1t, _ = eval_uf(self.model, self.type_vocab, test_batch_loader)
                    logging.info('TEST p={:.4f} r={:.4f} f1={:.4f}'.format(pt, rt, f1t))
                if f1 > best_dev_f1 and self.save_model_file is not None:
                    torch.save(self.model.state_dict(), self.save_model_file)
                    logging.info('model saved to {}'.format(self.save_model_file))
                if f1 > best_dev_f1:
                    best_dev_f1 = f1
                self.model.train()


def eval_uf(model, type_vocab, batch_iter, token_type=False, show_progress=False):
    from utils import utils

    results = list()
    gp_tups = list()
    logits_list, gold_tids_list = list(), list()
    for batch in batch_iter:
        with torch.no_grad():
            input_ids = batch['input_ids']
            attn_mask = batch['attn_mask']
            mask_idxs = batch['mask_idxs']
            token_type_ids = None
            if token_type:
                token_type_ids = batch['token_type_ids']
            logits_batch, _ = model(input_ids, attn_mask, mask_idxs, token_type_ids=token_type_ids)
            # logits_batch = model(token_id_seqs_tensor, attn_mask)
        logits_batch = logits_batch.data.cpu().numpy()

        gold_type_ids_list = batch['type_ids_list']
        for i, logits in enumerate(logits_batch):
            idxs = np.squeeze(np.argwhere(logits > 0), axis=1)
            if len(idxs) == 0:
                idxs = [np.argmax(logits)]
            logits_list.append(logits)

            gold_tids_list.append(gold_type_ids_list[i])
            gp_tups.append((gold_type_ids_list[i], idxs))
            r = {'types': [type_vocab[idx] for idx in idxs]}
            results.append(r)
            if show_progress and len(results) % 1000 == 0:
                print(len(results))
    p, r, f1 = utils.macro_f1_gptups(gp_tups)
    return p, r, f1, results
