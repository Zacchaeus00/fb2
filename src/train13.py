# train10 + linear all hidden layers (Model13)
import argparse
import os
import shutil
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from data_utils import FB2Dataset, prepare_data_token_cls
from eval_utils import eval_token_cls_model, convert_oof
from model_utils import Model13, strip_state_dict, process_state_dict
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
    arg('--lr_head', type=float, default=1e-4)
    arg('--seed', type=int, default=42)
    arg('--fold', default=0, type=int)
    arg('--exp', required=True, type=str)
    arg('--gradient_checkpointing', action="store_true", required=False)
    arg('--use_pretrained', type=str, default='')
    arg('--adv_lr', type=float, default=0)
    arg('--adv_eps', type=float, default=0.01)
    arg('--adv_after_epoch', type=float, default=0)
    arg('--only_infer', action="store_true", required=False)
    arg('--clip_grad_norm', type=float, default=10)
    arg('--patience', type=int, default=10)
    arg('--pooling', type=str, default='cls')
    arg('--reduction', type=str, default='mean')
    arg('--warmup_ratio', type=float, default=0)
    arg('--fix', action="store_true", required=False)
    arg('--hs_pooler_dropout', type=float, default=0.5)
    return parser.parse_args()


check_gpu()
cfg = parse_args_train()
print(datetime.datetime.now())
print(cfg)
seed = cfg.seed if cfg.seed != -1 else np.random.randint(0, 10000)
print("seed=", seed)
seed_everything(seed)
df = pd.read_csv('../data/train_folds.csv')
tokenizer = AutoTokenizer.from_pretrained(cfg.ckpt)
essay = pd.read_csv('../data/essay_processed.csv')
essay = essay.set_index('essay_id').squeeze()
train = pd.read_csv('../data/train_processed.csv')
samples = prepare_data_token_cls(essay, train, tokenizer, pooling=cfg.pooling, fix=cfg.fix, end=True)
print("samples[0]:", samples[0])
train_dataset = FB2Dataset([s for s in samples if s['fold'] != cfg.fold])
val_dataset = FB2Dataset([s for s in samples if s['fold'] == cfg.fold])
print(f"fold {cfg.fold}: n_train={len(train_dataset)}, n_val={len(val_dataset)}")
output_dir = f"../ckpt/{os.path.basename(__file__).split('.')[0]}/exp{cfg.exp}/"
Path(output_dir).mkdir(parents=True, exist_ok=True)

ckpt = cfg.use_pretrained if os.path.isdir(cfg.use_pretrained) else cfg.ckpt
print("ckpt:", ckpt)
model = Model13(ckpt,
                num_train_steps=int(len(train_dataset) / cfg.batch_size * cfg.epochs),
                lr=cfg.lr,
                lr_head=cfg.lr_head,
                reduction=cfg.reduction,
                warmup_ratio=cfg.warmup_ratio,
                hs_pooler_dropout=cfg.hs_pooler_dropout
                )
if cfg.use_pretrained and not os.path.isdir(cfg.use_pretrained):
    model.backbone.load_state_dict(strip_state_dict(torch.load(cfg.use_pretrained), cfg.ckpt), strict=False)
if cfg.gradient_checkpointing:
    model.backbone.gradient_checkpointing_enable()
model = Tez(model)
if not cfg.only_infer:
    es = EarlyStopping(
        monitor="valid_loss",
        model_path=os.path.join(output_dir, f"fold{cfg.fold}.pt"),
        patience=cfg.patience,
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
        clip_grad_norm=cfg.clip_grad_norm,
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
score, oof_df = eval_token_cls_model(model.model, [s for s in samples if s['fold'] == cfg.fold], pooling=cfg.pooling)
oof_df = convert_oof(oof_df)
oof_df.to_csv(os.path.join(output_dir, f"fold{cfg.fold}_oof.csv"), index=False)
print(f"fold {cfg.fold}: score={score}")
save_json({**vars(cfg), 'score': score, 'seed_used': seed}, os.path.join(output_dir, f"fold{cfg.fold}.json"))
get_cv(output_dir)
get_oof(output_dir)
