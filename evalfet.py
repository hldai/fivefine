import os
import torch
from tasks import fettask
from utils import utils
import config


def print_results(results):
    def mean(scores):
        return sum(scores) / len(scores)

    print()
    for r in results:
        print(r[0], r[1], r[2])
    print('AVG:', mean([r[0] for r in results]), mean([r[1] for r in results]), mean([r[2] for r in results]))


def eval_bbn():
    bbn_type_vocab_file = os.path.join(config.FET_DIR, 'alifet/bbn/bbn_type_vocab_ali.txt')
    data_file = os.path.join(config.FET_DIR, 'alifet/bbn/bbn_test_ali.json')
    load_model_files = [
        os.path.join(config.WORK_DIR, f'fet_models/tt_mlm_nw_bert_base_bbn_i{idx}.pth') for idx in range(0, 5)
    ]
    results = list()
    for load_model_file in load_model_files:
        predictor = fettask.FETPredictor(
            device, bbn_type_vocab_file, 'bbn', load_model_file,
            bert_model_str=config.BERT_BASE_MODEL_PATH, single_path=True)
        result = predictor.evaluate(data_file)
        results.append(result)
    print_results(results)


def eval_onto():
    onto_type_vocab_file = os.path.join(config.FET_DIR, 'alifet/ontonotes/onto_type_vocab_ali.txt')
    data_file = os.path.join(config.FET_DIR, 'alifet/ontonotes/onto_test_ali.json')
    load_model_files = [
        os.path.join(config.WORK_DIR, f'fet_models/tt_mlm_nw_bert_base_onto_i{idx}.pth') for idx in range(0, 5)
    ]
    results = list()
    for load_model_file in load_model_files:
        predictor = fettask.FETPredictor(
            device, onto_type_vocab_file, 'onto', load_model_file,
            bert_model_str=config.BERT_BASE_MODEL_PATH, single_path=True)
        result = predictor.evaluate(data_file)
        results.append(result)
    print_results(results)


def eval_fewnerd():
    fewnerd_type_vocab_file = os.path.join(config.FET_DIR, 'alifet/fewnerd/fewnerd_type_vocab.txt')
    data_file = os.path.join(config.FET_DIR, 'alifet/fewnerd/fewnerd_test_ali.json')
    load_model_files = [
        os.path.join(config.WORK_DIR, f'fet_models/tt_mlm_nw_bert_base_fewnerd_i{idx}.pth') for idx in range(0, 5)
    ]
    results = list()
    for load_model_file in load_model_files:
        predictor = fettask.FETPredictor(
            device, fewnerd_type_vocab_file, 'fewnerd', load_model_file,
            bert_model_str=config.BERT_BASE_MODEL_PATH, single_path=True)
        result = predictor.evaluate(data_file)
        results.append(result)
    print_results(results)


if __name__ == '__main__':
    args = utils.parse_idx_device_args()
    cuda_device_str = 'cuda' if len(args.d) == 0 else 'cuda:{}'.format(args.d[0])
    device = torch.device(cuda_device_str) if torch.cuda.device_count() > 0 else torch.device('cpu')

    if args.idx == 0:
        eval_fewnerd()
    if args.idx == 1:
        eval_onto()
    if args.idx == 2:
        eval_bbn()
