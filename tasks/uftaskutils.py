import torch
import numpy as np
from utils import utils


def eval_uf(model, type_vocab, batch_iter, token_type=False, show_progress=False):
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
