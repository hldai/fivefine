import logging
from functools import partial
import torch
import numpy as np
from transformers import BertTokenizer
from utils import utils, datautils, bertutils
from dataload import exampleload, batchload, dataloadutils
from models import modelutils
from models.bertet import TypeTokenBertET


class TrainConfig:
    def __init__(self, device, lr=3e-5, w_decay=0.01, batch_size=32, n_steps=2000, max_seq_len=128,
                 eval_interval=100, single_path_train=False, patience=500):
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.w_decay = w_decay
        self.max_seq_len = max_seq_len
        self.single_path_train = single_path_train
        self.n_steps = n_steps
        self.patience = patience
        self.eval_interval = eval_interval


def bert_batch_collect(device, type_id_dict, pad_token_id, examples):
    input_ids = [x['token_id_seq'] for x in examples]
    mask_idxs = [x['mask_idx'] for x in examples]
    input_ids, attn_mask = modelutils.pad_id_seqs(input_ids, device, pad_token_id)
    batch = {
        'input_ids': input_ids,
        'attn_mask': attn_mask,
        'mask_idxs': mask_idxs,
        'raw_examples': [x['raw'] for x in examples]
    }

    if len(examples) > 0 and 'labels' in examples[0] and type_id_dict is not None:
        labels_list = [x['labels'] for x in examples]
        type_ids_list = [[type_id_dict[t] for t in labels] for labels in labels_list]
        batch['type_ids_list'] = type_ids_list
        batch['labels_list'] = labels_list

    return batch


class DirectSingleTypeInfer:
    def __init__(self):
        pass

    def infer(self, logits_batch):
        idxs_list = list()
        for j, logits in enumerate(logits_batch):
            idxs = [np.argmax(logits)]
            idxs_list.append(idxs)
        return idxs_list


def spanm_tt_examples_iter(tokenizer, example_loader, max_seq_len, type_to_word_func):
    for i, x in enumerate(example_loader):
        bert_tok_id_seq, token_type_ids = dataloadutils.spanm_example_to_qic_bert_token_id_seq(
            tokenizer, x, max_seq_len)
        mask_idx = bert_tok_id_seq.index(tokenizer.mask_token_id)
        type_labels = x['labels']
        # type_words = [utils.fewnerd_type_to_word(t) for t in type_labels]
        type_words = [type_to_word_func(t) for t in type_labels]
        yield {
            'token_id_seq': bert_tok_id_seq,
            'mask_idx': mask_idx,
            'labels': type_words,
            'raw': x
        }


class FETModelTrainer:
    def __init__(
            self, tc: TrainConfig, type_vocab_file, dataset_name,
            tdt_files, load_model_file, bert_model_str, save_model_file=None,
            output_results_file=None
    ):
        self.tc = tc
        self.device = tc.device
        self.tdt_files = tdt_files
        raw_type_vocab, type_id_dict = datautils.load_vocab_file(type_vocab_file)
        self.dataset_name = dataset_name
        if dataset_name == 'onto':
            self.word_to_type_dict = utils.get_onto_word_to_type_dict(raw_type_vocab)
            self.type_to_word_func = utils.onto_type_to_word
        elif dataset_name == 'bbn':
            self.word_to_type_dict = utils.get_bbn_word_to_type_dict(raw_type_vocab, True)
            self.type_to_word_func = utils.bbn_type_to_word
        else:
            self.word_to_type_dict = utils.get_fewnerd_word_to_type_dict(raw_type_vocab)
            self.type_to_word_func = utils.fewnerd_type_to_word
        logging.info('{} types'.format(len(raw_type_vocab)))
        assert len(self.word_to_type_dict) == len(raw_type_vocab)
        self.word_type_vocab = list(self.word_to_type_dict.keys())
        self.word_type_id_dict = {wt: i for i, wt in enumerate(self.word_type_vocab)}
        self.n_types = len(self.word_type_vocab)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_str)

        self.pred_infer = DirectSingleTypeInfer()

        if load_model_file is None:
            self.model = TypeTokenBertET(bert_model_str, bert_model_str)
        else:
            logging.info('load_model_file={}'.format(load_model_file))
            self.model = TypeTokenBertET.from_trained(load_model_file, bert_model_str)
        self.model.to(self.device)
        self.model.init_type_hiddens(self.tokenizer, self.word_type_vocab)

        self.save_model_file = save_model_file
        self.output_results_file = output_results_file

    def run(self):
        self.model.train()
        logging.info(' '.join(['{}={}'.format(k, v) for k, v in vars(self.tc).items()]))
        tc = self.tc
        device = self.tc.device
        loss_obj = torch.nn.BCEWithLogitsLoss()
        optimizer = bertutils.get_bert_adam_optim(
            list(self.model.named_parameters()), learning_rate=tc.lr, w_decay=tc.w_decay)

        logging.info('train_file={}'.format(self.tdt_files['train']))
        logging.info('dev_file={}'.format(self.tdt_files['dev']))
        train_example_loader = exampleload.AliToSpanMExampleLoader(
            self.tdt_files['train'], remove_other=True, single_path=tc.single_path_train)
        dev_example_loader = exampleload.AliToSpanMExampleLoader(self.tdt_files['dev'])

        batch_collect_fn = partial(
            bert_batch_collect, self.device, self.word_type_id_dict, self.tokenizer.pad_token_id)

        test_file = self.tdt_files.get('test')
        logging.info('test_file={}'.format(test_file))
        test_batch_loader = None
        if test_file is not None:
            test_example_loader = exampleload.AliToSpanMExampleLoader(test_file)
            test_examples = list(spanm_tt_examples_iter(
                self.tokenizer, test_example_loader, tc.max_seq_len, self.type_to_word_func))
            test_batch_loader = batchload.IterExampleBatchLoader(
                test_examples, tc.batch_size, n_iter=1, collect_fn=batch_collect_fn)

        train_examples = list(spanm_tt_examples_iter(
            self.tokenizer, train_example_loader, tc.max_seq_len, self.type_to_word_func))
        dev_examples = list(spanm_tt_examples_iter(
            self.tokenizer, dev_example_loader, tc.max_seq_len, self.type_to_word_func))

        train_batch_loader = batchload.IterExampleBatchLoader(
            train_examples, tc.batch_size, n_steps=tc.n_steps * 5, collect_fn=batch_collect_fn)
        dev_batch_loader = batchload.IterExampleBatchLoader(
            dev_examples, tc.batch_size, n_iter=1, collect_fn=batch_collect_fn)

        step = 0
        last_best_step = 0
        losses = list()
        best_f1, best_acc = 0, 0
        best_test_result = None
        for batch in train_batch_loader:
            input_ids = batch['input_ids']
            attn_mask = batch['attn_mask']
            type_ids_list = batch['type_ids_list']
            mask_idxs = batch['mask_idxs']

            logits, _ = self.model(input_ids, attn_mask, mask_idxs)
            targets = modelutils.onehot_encode_batch(type_ids_list, self.n_types)
            targets = torch.tensor(targets, dtype=torch.float32, device=device)
            loss = loss_obj(logits, targets)

            losses.append(loss.data.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            # print(step)
            if step % tc.eval_interval == 0:
                self.model.eval()
                loss_val = sum(losses)
                losses = list()
                # print('eval ...')
                sacc, f1, mif1, dev_loss, _, _ = evaluate_fet_model(
                    self.model, dev_batch_loader, self.word_type_vocab, self.word_to_type_dict,
                    label_key='labels', pred_infer=self.pred_infer, loss_obj=loss_obj, device=tc.device)
                logging.info('i={} l={:.4f} ld={:.4f} acc={:.4f} maf1={:.4f} mif1={:.4f}'.format(
                    step, loss_val, dev_loss, sacc, f1, mif1))
                best_flg = f1 > best_f1 or (f1 == best_f1 and sacc > best_acc)
                if best_flg and test_batch_loader is not None and sacc > 0.3:
                    sacct, f1t, mif1t, dev_losst, _, pred_results = evaluate_fet_model(
                        self.model, test_batch_loader, self.word_type_vocab, self.word_to_type_dict,
                        label_key='labels', pred_infer=self.pred_infer)
                    logging.info('TEST acc={:.4f} maf1={:.4f} mif1={:.4f}'.format(sacct, f1t, mif1t))
                    best_test_result = (sacct, f1t, mif1t)

                if best_flg and self.save_model_file is not None:
                    torch.save(self.model.state_dict(), self.save_model_file)
                    logging.info('model save to {}'.format(self.save_model_file))
                if best_flg:
                    last_best_step = step
                    best_f1 = f1
                    best_acc = sacc
                self.model.train()

                if step >= self.tc.n_steps and step - last_best_step > tc.patience:
                    break
        return best_test_result


def predict_batches(model, batch_loader, word_type_vocab, pred_infer=None, loss_obj=None, device=None):
    for i, batch in enumerate(batch_loader):
        input_ids = batch['input_ids']
        attn_mask = batch['attn_mask']
        mask_idxs = batch['mask_idxs']
        loss_val = 0.0
        with torch.no_grad():
            logits_batch, _ = model(input_ids, attn_mask, mask_idxs)
            logits_batch_np = logits_batch.data.cpu().numpy()

            type_ids_list = batch.get('type_ids_list')
            if type_ids_list is not None and loss_obj is not None:
                targets = modelutils.onehot_encode_batch(type_ids_list, model.n_types)
                targets = torch.tensor(targets, dtype=torch.float32, device=device)
                loss = loss_obj(logits_batch, targets)
                loss_val = loss.data.cpu().numpy()
        raw_examples = batch['raw_examples']

        if pred_infer is None:
            for j, logits in enumerate(logits_batch_np):
                idxs = np.squeeze(np.argwhere(logits > 0), axis=1)
                if len(idxs) == 0:
                    idxs = [np.argmax(logits)]
                pred_word_labels = [word_type_vocab[idx] for idx in idxs]
                yield pred_word_labels, raw_examples[j], loss_val
        else:
            idxs_list = pred_infer.infer(logits_batch_np)
            for j in range(len(logits_batch_np)):
                pred_word_labels = [word_type_vocab[idx] for idx in idxs_list[j]]
                yield pred_word_labels, raw_examples[j], loss_val


def evaluate_fet_model(
        model, batch_loader, word_type_vocab, word_to_type_dict, label_key='y_str',
        pred_infer=None,
        full_labels=True,
        loss_obj=None,
        device=None,
):
    gp_pairs = list()
    losses = list()
    results = list()
    for i, (pred_label_words, raw_example, loss_val) in enumerate(
            predict_batches(model, batch_loader, word_type_vocab, pred_infer, loss_obj, device)):
        pred_labels = utils.get_fine_labels_from_words(
            pred_label_words, word_to_type_dict, get_full_type_labels=full_labels)
        results.append(pred_labels)
        true_labels = raw_example[label_key]
        true_labels = utils.get_full_types(true_labels)
        gp_pairs.append((true_labels, pred_labels))
        losses.append(loss_val)
        if (i + 1) % 10000 == 0:
            print(i + 1)
    sacc = utils.strict_acc_gp_pairs(gp_pairs)
    p, r, f1 = utils.macro_f1_gptups(gp_pairs)
    mif1 = utils.micro_f1(gp_pairs)
    return sacc, f1, mif1, sum(losses), len(gp_pairs), results
