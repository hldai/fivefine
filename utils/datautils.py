import json


def json_obj_iter(filename):
    with open(filename, encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def load_vocab_file(filename):
    with open(filename, encoding='utf-8') as f:
        vocab = [line.strip() for line in f]
    return vocab, {t: i for i, t in enumerate(vocab)}


def read_json_objs(filename):
    with open(filename, encoding='utf-8') as f:
        return [json.loads(line) for line in f]
