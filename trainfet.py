import datetime
import logging
import os
import torch
from tasks import fettask
from utils import utils
import config


def __setup_logging(to_file):
    log_file = os.path.join(config.WORK_DIR, 'log/{}-{}-{}.log'.format(os.path.splitext(
        os.path.basename(__file__))[0], args.idx, str_today)) if to_file else None
    utils.init_universal_logging(log_file, mode='a', to_stdout=True)
    logging.info('logging to {}'.format(log_file))


def train_bbn():
    __setup_logging(True)

    ali_bbn_type_vocab_file = os.path.join(config.FET_DIR, 'alifet/bbn/bbn_type_vocab_full.txt')
    load_model_file = os.path.join(config.WORK_DIR, 'uf_models/tt_mlm_nw_bert_base_best.pth')
    ali_bbn_tdt_data_files = dict()

    dataset_name = 'bbn'

    tc = fettask.TrainConfig(
        device=device,
        eval_interval=50,
        batch_size=48,
        lr=5e-6,
        single_path_train=True,
        n_steps=1000,
    )
    results = list()
    for i in range(0, 5):
        ali_bbn_tdt_data_files['train'] = os.path.join(
            config.FET_DIR, f'alifet/bbn/bbn_train_fewshot5_{i}.json')
        ali_bbn_tdt_data_files['dev'] = os.path.join(
            config.FET_DIR, f'alifet/bbn/bbn_dev_fewshot5_{i}.json')
        save_model_file = os.path.join(config.WORK_DIR, f'fet_models/tt_mlm_nw_bert_base_bbn_i{i}.pth')
        trainer = fettask.FETModelTrainer(
            tc, ali_bbn_type_vocab_file, dataset_name, ali_bbn_tdt_data_files, load_model_file,
            config.BERT_BASE_MODEL_PATH, save_model_file=save_model_file)
        r = trainer.run()
        if r is not None:
            results.append(r)
    if len(results) > 0:
        for r in results:
            print(r[0], r[1], r[2])


def train_onto():
    __setup_logging(True)

    ali_onto_type_vocab_file = os.path.join(config.FET_DIR, 'alifet/ontonotes/onto_type_vocab_ali.txt')
    load_model_file = os.path.join(config.WORK_DIR, 'uf_models/tt_mlm_nw_bert_base_best.pth')
    bert_model_str = config.BERT_BASE_MODEL_PATH
    ali_onto_tdt_data_files = dict()

    dataset_name = 'onto'

    tc = fettask.TrainConfig(
        device=device,
        eval_interval=10,
        batch_size=48,
        lr=1e-5,
        single_path_train=False,
        n_steps=1000,
        w_decay=0.01,
    )
    results = list()
    for i in range(0, 5):
        ali_onto_tdt_data_files['train'] = os.path.join(
            config.FET_DIR, f'alifet/onto/onto_train_fewshot5_{i}.json')
        ali_onto_tdt_data_files['dev'] = os.path.join(
            config.FET_DIR, f'alifet/onto/onto_dev_fewshot5_{i}.json')
        save_model_file = os.path.join(config.WORK_DIR, f'fet_models/tt_mlm_nw_bert_base_onto_i{i}.pth')
        logging.info(ali_onto_tdt_data_files['train'])
        trainer = fettask.FETModelTrainer(
            tc, ali_onto_type_vocab_file, dataset_name, ali_onto_tdt_data_files, load_model_file,
            bert_model_str, save_model_file=save_model_file)
        result = trainer.run()
        if result is not None:
            results.append(result)
    if len(results) > 0:
        for acc, maf1, mif1 in results:
            print('{} {} {}'.format(acc, maf1, mif1))


def train_fewnerd():
    __setup_logging(True)

    fewnerd_type_vocab_file = os.path.join(config.FET_DIR, 'alifet/fewnerd/fewnerd_type_vocab.txt')
    load_model_file = os.path.join(config.WORK_DIR, 'uf_models/tt_mlm_nw_bert_base_best.pth')
    save_model_file = None
    bert_model_str = config.BERT_BASE_MODEL_PATH
    fewnerd_tdt_data_files = dict()

    dataset_name = 'fewnerd'
    tc = fettask.TrainConfig(
        device=device,
        eval_interval=50,
        batch_size=64,
        lr=1e-5,
        n_steps=1000,
        patience=500,
    )

    for i in range(5):
        fewnerd_tdt_data_files['train'] = os.path.join(
            config.FET_DIR, f'alifet/fewnerd/fewnerd_train_fewshot5_{i}.json')
        fewnerd_tdt_data_files['dev'] = os.path.join(
            config.FET_DIR, f'alifet/fewnerd/fewnerd_dev_fewshot5_{i}.json')
        save_model_file = os.path.join(config.WORK_DIR, f'fet_models/tt_mlm_nw_bert_base_fewnerd_i{i}.pth')
        trainer = fettask.FETModelTrainer(
            tc, fewnerd_type_vocab_file, dataset_name, fewnerd_tdt_data_files, load_model_file,
            bert_model_str, save_model_file=save_model_file)
        trainer.run()


def train_manual_onto():
    print('train_manual_onto')
    onto_type_vocab_file = os.path.join(config.FET_DIR, 'ontonotes/onto_ontology.txt')
    manual_onto_data_files = {
        'train': os.path.join(config.FET_DIR, 'ontonotes/onto_anno_full_10.json'),
        'dev': os.path.join(config.FET_DIR, 'ontonotes/g_dev_dev_5.json'),
        'test': os.path.join(config.FET_DIR, 'ontonotes/g_test.json'),
    }
    load_model_file = os.path.join(config.WORK_DIR, 'uf_models/tt_mlm_nw_bert_base-[best].pth')
    # save_model_file = os.path.join(config.WORK_DIR, 'fet_models/tt_mlm_nw_qic_base_manonto.pth')
    save_model_file = None
    tc = fettask.TrainConfig(
        device=device,
        batch_size=64,
        w_decay=0.01,
        lr=3e-5,
        n_steps=1200,
        lr_schedule=True,
    )
    trainer = fettask.ManualOntoTrainer(
        tc, onto_type_vocab_file, manual_onto_data_files, load_model_file, config.BERT_BASE_MODEL_PATH,
        save_model_file=save_model_file
    )
    trainer.run()


if __name__ == '__main__':
    utils.set_all_random_seed(7771)
    str_today = datetime.date.today().strftime('%y-%m-%d')
    args = utils.parse_idx_device_args()
    cuda_device_str = 'cuda' if len(args.d) == 0 else 'cuda:{}'.format(args.d[0])
    device = torch.device(cuda_device_str) if torch.cuda.device_count() > 0 else torch.device('cpu')
    device_ids = args.d
    utils.init_universal_logging(None)

    if args.idx == 0:
        train_fewnerd()
    elif args.idx == 1:
        train_onto()
    elif args.idx == 2:
        train_bbn()
    elif args.idx == 3:
        train_manual_onto()
