# train3 + different lr
import argparse
import os
import shutil
import datetime

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForTokenClassification, get_linear_schedule_with_warmup

from data_utils import FB2Dataset, prepare_data_token_cls
from eval_utils import eval_token_cls_model
from model_utils import Model3, strip_state_dict
from utils import seed_everything, save_json, get_cv, get_oof, check_gpu

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def parse_args_train():
    parser = argparse.ArgumentParser(description='')
    arg = parser.add_argument
    arg('--ckpt', type=str, default='/gpfsnyu/scratch/yw3642/hf-models/microsoft_deberta-v3-base')
    arg('--epochs', type=int, default=5)
    arg('--batch_size', type=int, default=1)
    arg('--lr', type=float, default=1e-5)
    arg('--lr_head', type=float, default=1e-3)
    arg('--weight_decay', type=float, default=0)
    arg('--seed', type=int, default=42)
    arg('--fold', default=0, type=int)
    arg('--exp', required=True, type=int)
    arg('--gradient_checkpointing', action="store_true", required=False)
    arg('--use_pretrained', type=str, default='')
    arg('--only_infer', action="store_true", required=False)
    arg('--use_stable_dropout', action="store_true", required=False)
    arg('--normal_init', action="store_true", required=False)
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
model = Model3(cfg.ckpt, use_stable_dropout=cfg.use_stable_dropout, normal_init=cfg.normal_init)
if cfg.use_pretrained:
    model.backbone.load_state_dict(strip_state_dict(torch.load(cfg.use_pretrained), cfg.ckpt), strict=True)
if cfg.gradient_checkpointing:
    model.gradient_checkpointing_enable()

train_samples = [s for s in samples if s['fold'] != cfg.fold]
val_samples = [s for s in samples if s['fold'] == cfg.fold]
print(f"fold {cfg.fold}: n_train={len(train_samples)}, n_val={len(val_samples)}")
output_dir = f"../ckpt/{os.path.basename(__file__).split('.')[0]}/exp{cfg.exp}/"
if not cfg.only_infer:
    args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"fold{cfg.fold}"),
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
        logging_steps=1000,
        save_steps=1000,
        eval_steps=1000,
    )
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
            "lr": cfg.lr,
        },
        {
            "params": [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": cfg.lr,
        },
        {
            "params": [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
            "lr": cfg.lr_head,
        },
        {
            "params": [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": cfg.lr_head,
        },
    ]
    optimizer = AdamW(optimizer_parameters)
    train_steps = len(train_samples) * cfg.epochs // cfg.batch_size
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.2 * train_steps,
        num_training_steps=train_steps,
        last_epoch=-1,
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=FB2Dataset(train_samples),
        eval_dataset=FB2Dataset(val_samples),
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        optimizers=[optimizer, scheduler]
    )
    trainer.train()
    torch.save(model.state_dict(), os.path.join(output_dir, f"fold{cfg.fold}.pt"))
    shutil.rmtree(os.path.join(output_dir, f"fold{cfg.fold}"))
else:
    model.load_state_dict(torch.load(os.path.join(output_dir, f"fold{cfg.fold}.pt")))
score, oof_df = eval_token_cls_model(model, val_samples)
oof_df.to_pickle(os.path.join(output_dir, f"fold{cfg.fold}_oof.gz"))
print(f"fold {cfg.fold}: score={score}")
save_json({**vars(cfg), 'score': score}, os.path.join(output_dir, f"fold{cfg.fold}.json"))
get_cv(output_dir)
get_oof(output_dir)
