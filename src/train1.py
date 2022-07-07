# train0 + reinit
import argparse
import os
import shutil
import datetime

import numpy as np
import pandas as pd
import sklearn
import torch
from torch.utils.checkpoint import checkpoint
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from data_utils import prepare_data_mp, FB2Dataset
from utils import seed_everything, save_json, get_cv
# from model_utils import reinit_deberta_v2_pooler, reinit_deberta_v2_layers

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def parse_args_train():
    parser = argparse.ArgumentParser(description='')
    arg = parser.add_argument
    arg('--ckpt', type=str, default='/gpfsnyu/scratch/yw3642/hf-models/microsoft_deberta-v3-large')
    arg('--epochs', type=int, default=5)
    arg('--batch_size', type=int, default=2)
    arg('--lr', type=float, default=1e-5)
    arg('--weight_decay', type=float, default=0)
    arg('--seed', type=int, default=42)
    arg('--fold', default=0, type=int)
    arg('--max_len', default=1024, type=int)
    arg('--exp', required=True, type=int)
    arg('--gradient_checkpointing', action="store_true", required=False)
    arg('--reinit_pooler', action="store_true", required=False)
    arg('--reinit_layers', default=0, type=int)
    return parser.parse_args()


cfg = parse_args_train()
print(datetime.datetime.now())
print(cfg)
seed_everything(cfg.seed)
df = pd.read_csv('../data/train_folds.csv')
tokenizer = AutoTokenizer.from_pretrained(cfg.ckpt)
samples = prepare_data_mp('../data/feedback-prize-effectiveness/train', tokenizer, df, max_len=cfg.max_len, j=8)
model = AutoModelForSequenceClassification.from_pretrained(cfg.ckpt, num_labels=3)
if cfg.gradient_checkpointing:
    model.gradient_checkpointing_enable()
if cfg.reinit_pooler:
    reinit_deberta_v2_pooler(model)
if cfg.reinit_layers > 0:
    reinit_deberta_v2_layers(model, n=cfg.reinit_layers)

train_samples = [s for s in samples if s['fold'] != cfg.fold]
val_samples = [s for s in samples if s['fold'] == cfg.fold]
print(f"fold {cfg.fold}: n_train={len(train_samples)}, n_val={len(val_samples)}")
args = TrainingArguments(
    output_dir=f"../ckpt/train1/exp{cfg.exp}/fold{cfg.fold}",
    save_strategy="steps",
    evaluation_strategy="steps",
    learning_rate=cfg.lr,
    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=8,
    num_train_epochs=cfg.epochs,
    weight_decay=cfg.weight_decay,
    warmup_ratio=0.2,
    fp16=True,
    report_to='none',
    dataloader_num_workers=4,
    load_best_model_at_end=True,
    group_by_length=True,
    save_total_limit=1,
    seed=cfg.seed,
    logging_steps=2500,
    save_steps=1500,
    eval_steps=1500,
)
trainer = Trainer(
    model,
    args,
    train_dataset=FB2Dataset(train_samples),
    eval_dataset=FB2Dataset(val_samples),
    tokenizer=tokenizer,
)
trainer.train()
predictions = trainer.predict(FB2Dataset(val_samples)).predictions
np.save(f"../ckpt/train1/exp{cfg.exp}/oof_logits_fold{cfg.fold}.npy", predictions)
probs = softmax(predictions, axis=1)
probs = np.clip(probs, 1e-15, 1 - 1e-15)
score = sklearn.metrics.log_loss([s['label'] for s in val_samples], probs)
print(f"fold {cfg.fold}: score={score}")

torch.save(model.state_dict(), f"../ckpt/train1/exp{cfg.exp}/fold{cfg.fold}.pt")
shutil.rmtree(f"../ckpt/train1/exp{cfg.exp}/fold{cfg.fold}")
save_json({**vars(cfg), 'score': score}, f"../ckpt/train1/exp{cfg.exp}/fold{cfg.fold}.json")
get_cv(f"../ckpt/train1/exp{cfg.exp}/")
