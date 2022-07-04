# train3 + tez
import argparse
import os
import shutil
import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from data_utils import FB2Dataset, prepare_data_token_cls
from eval_utils import eval_token_cls_model
from model_utils import Model5, strip_state_dict
from utils import seed_everything, save_json, get_cv, get_oof, check_gpu
from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def parse_args_train():
    parser = argparse.ArgumentParser(description='')
    arg = parser.add_argument
    arg('--ckpt', type=str, default='microsoft/deberta-v3-base')
    arg('--epochs', type=int, default=10)
    arg('--batch_size', type=int, default=2)
    arg('--lr', type=float, default=2e-5)
    arg('--seed', type=int, default=42)
    arg('--fold', default=0, type=int)
    arg('--exp', default=-1, type=int)
    arg('--gradient_checkpointing', action="store_true", required=False)
    arg('--use_pretrained', type=str, default='')
    arg('--adv_lr', type=float, default=0)
    arg('--adv_eps', type=float, default=0.01)
    arg('--adv_after_epoch', type=float, default=0)
    arg('--only_infer', action="store_true", required=False)
    return parser.parse_args()


check_gpu()
cfg = parse_args_train()
print(datetime.datetime.now())
print(cfg)
seed_everything(cfg.seed)
df = pd.read_csv('../data/train_folds.csv')
tokenizer = AutoTokenizer.from_pretrained(cfg.ckpt)
essay = pd.read_csv('../data/essay_processed.csv')
essay = essay.set_index('essay_id').squeeze()
train = pd.read_csv('../data/train_processed.csv')
samples = prepare_data_token_cls(essay, train, tokenizer)
train_dataset = FB2Dataset([s for s in samples if s['fold'] != cfg.fold])
val_dataset = FB2Dataset([s for s in samples if s['fold'] == cfg.fold])
print(f"fold {cfg.fold}: n_train={len(train_dataset)}, n_val={len(val_dataset)}")
output_dir = f"../ckpt/{os.path.basename(__file__).split('.')[0]}/exp{cfg.exp}/"
Path(output_dir).mkdir(parents=True, exist_ok=True)

model = Model5(cfg.ckpt,
               num_train_steps=int(len(train_dataset) / cfg.batch_size * cfg.epochs),
               learning_rate=cfg.lr)
if cfg.use_pretrained:
    model.backbone.load_state_dict(strip_state_dict(torch.load(cfg.use_pretrained), cfg.ckpt), strict=True)
if cfg.gradient_checkpointing:
    model.backbone.gradient_checkpointing_enable()
model = Tez(model)
if not cfg.only_infer:
    es = EarlyStopping(
        monitor="valid_loss",
        model_path=os.path.join(output_dir, f"fold{cfg.fold}.pt"),
        patience=5,
        mode="min",
        delta=0.001,
        save_weights_only=True,
    )
    config = TezConfig(
        training_batch_size=cfg.batch_size,
        validation_batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        fp16=True,
        step_scheduler_after="batch",
        val_strategy="batch",
        val_steps=500,
        adv_lr=cfg.adv_lr,
        adv_eps=cfg.adv_eps,
        adv_after_epoch=cfg.adv_after_epoch,
        valid_shuffle=False,
    )
    model.fit(
        train_dataset,
        valid_dataset=val_dataset,
        train_collate_fn=DataCollatorForTokenClassification(tokenizer),
        valid_collate_fn=DataCollatorForTokenClassification(tokenizer),
        callbacks=[es],
        config=config,
    )
model.model.load_state_dict(torch.load(os.path.join(output_dir, f"fold{cfg.fold}.pt")))
score, oof_df = eval_token_cls_model(model.model, [s for s in samples if s['fold'] == cfg.fold])
print(f"fold {cfg.fold}: score={score}")
save_json({**vars(cfg), 'score': score}, os.path.join(output_dir, f"fold{cfg.fold}.json"))
get_cv(output_dir)
get_oof(output_dir)
