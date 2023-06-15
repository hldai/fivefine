import logging


def init_universal_logging(logfile=None, mode='a', to_stdout=True, use_color=True):
    handlers = list()
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode=mode))
    if to_stdout:
        sh = logging.StreamHandler()
        if use_color:
            prefix_color = '\x1b[34m'
            # prefix_color = '\x1b[90m'
            bold = '\x1b[1m'
            reset = '\x1b[0m'
            colored_formatter = logging.Formatter(
                fmt=f'{prefix_color}%(asctime)s %(filename)s:%(lineno)s {bold}%(levelname)s{reset}{prefix_color} - '
                    f'{reset}%(message)s',
                datefmt='%y-%m-%d %H:%M:%S')
            sh.setFormatter(colored_formatter)
        handlers.append(sh)
    logging.basicConfig(format='%(asctime)s %(filename)s:%(lineno)s %(levelname)s - %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S', handlers=handlers, level=logging.INFO)
    logging.info('logging to {}'.format(logfile))


def set_all_random_seed(seed_val):
    import random
    import numpy as np
    import torch

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)


def parse_idx_device_args(arg_str: str = None):
    import argparse

    parser = argparse.ArgumentParser(description='dhl')
    parser.add_argument('idx', type=int, default=0, nargs='?')
    parser.add_argument('-d', type=int, default=[], nargs='+')
    if arg_str is not None and len(arg_str) > 0:
        return parser.parse_args(arg_str.split(' '))
    return parser.parse_args()


def calc_f1(p, r):
    if r == 0.:
        return 0.
    return 2 * p * r / float(p + r)


def macro_f1_gptups(true_and_prediction):
    # num_examples = len(true_and_prediction)
    p, r = 0., 0.
    pred_example_count, gold_example_count = 0., 0.
    pred_label_count, gold_label_count = 0., 0.
    for true_labels, predicted_labels in true_and_prediction:
        # print(predicted_labels)
        if len(predicted_labels) > 0:
            pred_example_count += 1
            pred_label_count += len(predicted_labels)
            per_p = len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
            p += per_p
        if len(true_labels) > 0:
            gold_example_count += 1
            per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
            r += per_r
    precision, recall = 0, 0
    if pred_example_count > 0:
        precision = p / pred_example_count
    if gold_example_count > 0:
        recall = r / gold_example_count
    # avg_elem_per_pred = pred_label_count / pred_example_count
    return precision, recall, calc_f1(precision, recall)
