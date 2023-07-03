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
