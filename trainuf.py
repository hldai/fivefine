import datetime
import logging
import os
import torch
import config
from utils import utils
from tasks import uftask, ufsttask


type_vocab_file = os.path.join(config.UF_DIR, 'ontology/types.txt')

uf_train_file = os.path.join(config.UF_DIR, 'crowd/train.json')
uf_dev_file = os.path.join(config.UF_DIR, 'crowd/dev.json')
uf_test_file = os.path.join(config.UF_DIR, 'crowd/test.json')

uf_tdt_files = {
    'train': uf_train_file,
    'dev': uf_dev_file,
    'test': uf_test_file,
}

uf_data_files = {
    'el_train_file': os.path.join(config.UF_DIR, 'uf_total_train/el_train.json'),
    'el_extra_label_file': os.path.join(config.UF_DIR, 'bert_labels/el_train_ama_ms_10types.json'),
    'open_train_files': [os.path.join(
        config.UF_DIR, 'uf_total_train/open_train_{:02d}.json'.format(i)) for i in range(21)],
    'open_extra_label_files': [os.path.join(
        config.UF_DIR,
        'bert_labels/open_train_{:02d}_ama_ms_10types.json'.format(i)) for i in range(21)],
    'pronoun_mention_file': os.path.join(
        config.UF_DIR, 'pronoun_data/gigaword_eng_5_texts_pronoun_s005.txt'),
    'pronoun_label_file': os.path.join(
        config.UF_DIR, 'bert_labels/gigaword5_pronoun_s005_ama_ms_10types.json'),
    'dev': uf_dev_file,
}


def __setup_logging(to_file):
    log_file = os.path.join(config.WORK_DIR, 'log/{}-{}-{}.log'.format(os.path.splitext(
        os.path.basename(__file__))[0], args.idx, str_today)) if to_file else None
    utils.init_universal_logging(log_file, mode='a', to_stdout=True)
    logging.info('logging to {}'.format(log_file))


def train_uf_st():
    __setup_logging(True)
    bert_model_str = config.BERT_BASE_MODEL_PATH
    load_model_file = os.path.join(config.WORK_DIR, 'uf_models/tt_mlm_nw_bert_base-[best].pth')
    teacher_model_file = os.path.join(config.WORK_DIR, 'uf_models/tt_mlm_nw_bert_base_ft.pth')

    save_model_file = os.path.join(config.WORK_DIR, 'uf_models/tt_mlm_nw_bert_base_ftst_r{}.pth')

    tc = ufsttask.TrainConfig(
        device,
        batch_size=128,
        gacc_step=1,
        lr=1e-5,
        eval_interval=100,
        model_str=bert_model_str,
        teacher_model_str=config.BERT_BASE_MODEL_PATH,
        n_steps=200000,
        w_decay=0.1,
        src_load_weights=(0.4, 0.2, 0.2, 0.2)
    )

    trainer = ufsttask.UFSTTrainer(
        tc, type_vocab_file, uf_tdt_files, teacher_model_file, uf_data_files,
        load_model_file=load_model_file,
        save_model_file=save_model_file
    )
    trainer.train()


def finetune_uf():
    __setup_logging(True)
    load_model_file = os.path.join(config.WORK_DIR, 'uf_models/tt_mlm_nw_bert_base-[best].pth')
    save_model_file = os.path.join(config.WORK_DIR, 'uf_models/tt_mlm_nw_bert_base_ft.pth')
    tc = uftask.TrainConfig(device, lr=1e-5, batch_size=32, n_steps=5000, eval_interval=100)
    trainer = uftask.TypeTokenModelTrainer(
        tc, config.BERT_BASE_MODEL_PATH, type_vocab_file, uf_tdt_files,
        load_model_file=load_model_file, save_model_file=save_model_file)
    trainer.train()


def train_uf():
    # __setup_logging(False)
    __setup_logging(True)
    tc = uftask.TrainConfig(
        device,
        batch_size=64,
        gacc_step=1,
        eval_interval=2000,
        save_interval=100000,
        n_steps=1000000,
        mlm_lamb=0.1,
        lr=1e-5,
        n_ex_types=10,
        weight_for_origin_label=5.0,
        max_seq_len=128,
        neighbor_words=True,
        nw_train_interval=10,
        nw_lamb=0.1,
    )

    model_str = config.BERT_BASE_MODEL_PATH
    trainer = uftask.WeakUFModelTrainer(
        tc, type_vocab_file, model_str, uf_data_files,
        load_model_file=None,
        save_model_file_prefix=os.path.join(config.WORK_DIR, 'uf_models/tt_mlm_nw_bert_base'),
        # save_model_file_prefix=None,
    )
    trainer.train()


if __name__ == '__main__':
    utils.set_all_random_seed(7771)
    str_today = datetime.date.today().strftime('%y-%m-%d')
    args = utils.parse_idx_device_args()
    cuda_device_str = 'cuda' if len(args.d) == 0 else 'cuda:{}'.format(args.d[0])
    device = torch.device(cuda_device_str) if torch.cuda.device_count() > 0 else torch.device('cpu')
    device_ids = args.d

    if args.idx == 0:
        train_uf()
    elif args.idx == 1:
        finetune_uf()
    elif args.idx == 2:
        train_uf_st()
