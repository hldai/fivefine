import torch
import numpy as np


def pad_id_seqs(id_seqs, device, pad_id, fixed_max_len=-1, pad_before=False):
    batch_size = len(id_seqs)
    max_len = max(len(seq) for seq in id_seqs) if fixed_max_len < 0 else fixed_max_len
    id_seqs_arr = np.ones((batch_size, max_len), dtype=np.int32) * pad_id
    for i, seq in enumerate(id_seqs):
        # print(len(seq), max_len)
        if pad_before:
            id_seqs_arr[i][-len(seq):] = seq
        else:
            id_seqs_arr[i][:len(seq)] = seq
    id_seqs_tensor = torch.tensor(id_seqs_arr, dtype=torch.long, device=device)
    attn_mask = (id_seqs_tensor != pad_id).to(torch.long)
    return id_seqs_tensor, attn_mask


def pad_seq_to_len(ids_list, target_len, device):
    batch_size = len(ids_list)
    ids_arr = np.zeros((batch_size, target_len), dtype=np.float32)
    for i, ids in enumerate(ids_list):
        for j in range(min(len(ids), target_len)):
            ids_arr[i][j] = ids[j]
        # for j, id_val in enumerate(ids):
        #     ids_arr[i][j] = id_val
    id_seqs_tensor = torch.tensor(ids_arr, dtype=torch.long, device=device)
    return id_seqs_tensor


def onehot_encode_batch(class_ids_list, n_classes):
    batch_size = len(class_ids_list)
    tmp = np.zeros((batch_size, n_classes), dtype=np.float32)
    for i, class_ids in enumerate(class_ids_list):
        for cid in class_ids:
            tmp[i][cid] = 1.0
    return tmp


def save_model(model, model_file, is_parallel):
    import logging

    if is_parallel:
        torch.save(model.module.state_dict(), model_file)
    else:
        torch.save(model.state_dict(), model_file)
    logging.info('model saved to {}'.format(model_file))
