# nbroad + Model8
import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping
from torch.utils.checkpoint import checkpoint
from transformers import DataCollatorForTokenClassification

from model_utils import Model8, strip_state_dict
from nbroad import get_dataset
from eval_utils import eval_token_cls_model, convert_oof
from utils import seed_everything, check_gpu, save_json, get_cv, get_oof

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
    arg('--resize_embedding', action="store_true", required=False)
    arg('--random_fold', action="store_true", required=False)
    return parser.parse_args()


check_gpu()
cfg = parse_args_train()
print(datetime.datetime.now())
print(cfg)
seed = cfg.seed if cfg.seed != -1 else np.random.randint(0, 10000)
print("seed=", seed)
seed_everything(seed)
nb_cfg = {
    'num_proc': 1,
    "data_dir": "../data/feedback-prize-effectiveness",
    "model_name_or_path": cfg.ckpt,
}
ds, tokenizer = get_dataset(nb_cfg)
assert '[CLS_CLAIM]' in tokenizer.get_vocab()
assert '[END_CLAIM]' in tokenizer.get_vocab()
print("dataset:", ds)
print("dataset[0]:", ds[0])


if cfg.random_fold:
    print('RANDOM FOLD')
    folds = np.random.randint(0, 5, len(ds))
else:
    fold_df = pd.read_csv('../data/train_folds.csv')
    folds = []
    for sample in ds:
        eid = sample['essay_id']
        df = fold_df[fold_df['essay_id']==eid]
        assert(df['kfold'].nunique()==1)
        folds.append(df['kfold'].values[0])
ds = ds.add_column("fold", folds)

keep_cols = {"input_ids", "attention_mask", "labels"}
train_idxs = [i for i, sample in enumerate(ds) if sample['fold'] != cfg.fold]
eval_idxs = [i for i, sample in enumerate(ds) if sample['fold'] == cfg.fold]
train_dataset = ds.select(train_idxs).remove_columns([c for c in ds.column_names if c not in keep_cols])
eval_dataset = ds.select(eval_idxs)
val_dataset = eval_dataset.remove_columns([c for c in ds.column_names if c not in keep_cols])
print(f"fold {cfg.fold}: n_train={len(train_dataset)}, n_val={len(val_dataset)}")
output_dir = f"../ckpt/{os.path.basename(__file__).split('.')[0]}/exp{cfg.exp}/"
Path(output_dir).mkdir(parents=True, exist_ok=True)

ckpt = cfg.use_pretrained if os.path.isdir(cfg.use_pretrained) else cfg.ckpt
print("ckpt:", ckpt)
model = Model8(ckpt,
               num_train_steps=int(len(train_dataset) / cfg.batch_size * cfg.epochs),
               lr=cfg.lr,
               lr_head=cfg.lr_head,
               reduction=cfg.reduction,
               warmup_ratio=cfg.warmup_ratio
               )
if cfg.use_pretrained and not os.path.isdir(cfg.use_pretrained):
    model.backbone.load_state_dict(strip_state_dict(torch.load(cfg.use_pretrained), cfg.ckpt), strict=False)
if cfg.gradient_checkpointing:
    model.backbone.gradient_checkpointing_enable()
if cfg.resize_embedding:
    print("resize embedding to len(tokenizer) =", len(tokenizer))
    model.backbone.resize_token_embeddings(len(tokenizer))
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
score, oof_df = eval_token_cls_model(model.model, eval_dataset, pooling=cfg.pooling)
oof_df = convert_oof(oof_df)
oof_df.to_csv(os.path.join(output_dir, f"fold{cfg.fold}_oof.csv"), index=False)
print(f"fold {cfg.fold}: score={score}")
save_json({**vars(cfg), 'score': score, 'seed_used': seed}, os.path.join(output_dir, f"fold{cfg.fold}.json"))
get_cv(output_dir)
get_oof(output_dir)
