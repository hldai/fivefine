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


def onto_type_to_word(type_name):
    if type_name == '/person/artist/music':
        return 'musician'
    if type_name == '/location/structure/government':
        return 'government_location'
    if type_name == '/organization/military':
        return 'military_organization army'
    if type_name == '/person/legal':
        return 'legal_person lawyer judge'
    if type_name == '/organization/transit':
        return 'transit_organization'
    if type_name == '/organization/company/broadcast':
        return 'broadcast_company television_network'
    if type_name == '/organization/music':
        return 'music_organization'
    p = type_name.rfind('/')
    w = type_name[p + 1:]
    return w


def get_onto_word_to_type_dict(type_vocab):
    word_to_type_dict = dict()
    for type_name in type_vocab:
        w = onto_type_to_word(type_name)
        word_to_type_dict[w] = type_name
    word_to_type_dict['island'] = '/location/geography/island'
    return word_to_type_dict


def bbn_type_to_word(type_name, underscore_to_space=True):
    type_word = ''
    if type_name == '/GPE':
        type_word = 'geo_political'
    elif type_name == '/FAC':
        type_word = 'facility'
    elif type_name == '/ORGANIZATION/CORPORATION':
        type_word = 'company'
    else:
        type_name_lower = type_name.lower()

        p = type_name_lower.rfind('/')
        type_word = type_name_lower[p + 1:]

    if underscore_to_space:
        type_word = type_word.replace('_', ' ')
    return type_word


def get_bbn_word_to_type_dict(type_vocab, underscore_to_space):
    word_to_type_dict = dict()
    for type_name in type_vocab:
        w = bbn_type_to_word(type_name, underscore_to_space)
        # print(type_name, w)
        word_to_type_dict[w] = type_name
    return word_to_type_dict


def fewnerd_type_to_word(type_name):
    if type_name == '/art/writtenart':
        type_word = 'written work book newspaper'
    elif type_name == '/government/government':
        type_word = 'government organization'
    elif type_name == '/organization/media_newspaper':
        type_word = 'newspaper company'
    elif type_name == '/organization/sportsteam':
        type_word = 'sports team'
    elif type_name == '/organization/sportsleague':
        type_word = 'sports league'
    elif type_name == '/organization/government_governmentagency':
        type_word = 'government agency'
    elif type_name == '/other/biologything':
        type_word = 'biology'
    elif type_name == '/location/GPE':
        type_word = 'location city town state country'
    elif type_name == '/art/broadcastprogram':
        type_word = 'broadcast program'
    elif type_name == '/building/sportsfacility':
        type_word = 'sports facility'
    elif type_name == '/other/chemicalthing':
        type_word = 'chemical'
    elif type_name == '/location/bodiesofwater':
        type_word = 'lake river sea'
    elif type_name == '/other/astronomything':
        type_word = 'astronomy'
    elif type_name == '/event/sportsevent':
        type_word = 'sports game match'
    elif type_name == '/organization/showorganization':
        type_word = 'band'
    elif type_name == '/other/educationaldegree':
        type_word = 'educational degree'
    elif type_name == '/organization/politicalparty':
        type_word = 'political party'
    elif type_name == '/event/attack_battle_war_militaryconflict':
        type_word = 'attack battle war military conflict'
    elif type_name == '/other/livingthing':
        type_word = 'living thing'
    elif type_name == '/organization/education':
        type_word = 'education school university'
    else:
        p = type_name.rfind('/')
        type_word = type_name[p + 1:]
        if type_word == 'other':
            type_word = type_name.split('/')[1]

    type_word = type_word.replace('_', ' ')
    return type_word


def get_fewnerd_word_to_type_dict(type_vocab):
    word_to_type_dict = dict()
    for type_name in type_vocab:
        w = fewnerd_type_to_word(type_name)
        word_to_type_dict[w] = type_name
    return word_to_type_dict


def __super_types(t):
    types = [t]
    tmpt = t
    while True:
        pos = tmpt.rfind('/')
        if pos == 0:
            break
        tmpt = tmpt[:pos]
        types.append(tmpt)
    return types


def get_full_types(labels):
    types = set()
    for label in labels:
        super_types = __super_types(label)
        for t in super_types:
            types.add(t)
    return list(types)


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


def get_fine_labels_from_words(type_words, word_to_type_dict, get_full_type_labels=True):
    labels = list()
    for w in type_words:
        t = word_to_type_dict.get(w, None)
        if t is not None:
            labels.append(t)
    if get_full_type_labels:
        labels = get_full_types(labels)
    return labels


def strict_match(labels_true, labels_pred):
    if len(labels_pred) != len(labels_true):
        return False
    for lt in labels_true:
        if lt not in labels_pred:
            return False
    return True


def strict_acc_gp_pairs(gp_tups):
    hit_cnt = sum(1 if strict_match(labels_true, labels_pred) else 0 for labels_true, labels_pred in gp_tups)
    return hit_cnt / (len(gp_tups) + 0.0001)


def count_match(label_true, label_pred):
    return sum(1 if t in label_pred else 0 for t in label_true)


def micro_f1(gold_pred_label_pairs):
    l_true_cnt, l_pred_cnt, hit_cnt = 0, 0, 0
    for labels_true, labels_pred in gold_pred_label_pairs:
        hit_cnt += count_match(labels_true, labels_pred)
        l_true_cnt += len(labels_true)
        l_pred_cnt += len(labels_pred)
    p = hit_cnt / l_pred_cnt
    r = hit_cnt / l_true_cnt
    return 2 * p * r / (p + r + 1e-7)
