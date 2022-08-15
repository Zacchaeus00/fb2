# pretrain0 + raw text for pretrain
import argparse
import shutil
import os
import datetime

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForWholeWordMask
from torch.utils.checkpoint import checkpoint

from data_utils import PretrainDataset1
from utils import save_json

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def parse_args_pretrain():
    parser = argparse.ArgumentParser(description='')
    arg = parser.add_argument
    arg('--ckpt', type=str, default='/gpfsnyu/scratch/yw3642/hf-models/microsoft_deberta-v3-large')
    arg('--epochs', type=int, default=5)
    arg('--batch_size', type=int, default=2)
    arg('--lr', type=float, default=1e-5)
    arg('--weight_decay', type=float, default=0)
    arg('--seed', type=int, default=42)
    arg('--exp', required=True, type=int)
    arg('--max_len', type=int, default=1024)
    arg('--gradient_checkpointing', action="store_true", required=False)
    arg('--mlm_prob', type=float, default=0.15)
    arg('--resize_embedding', action="store_true", required=False)
    return parser.parse_args()


cfg = parse_args_pretrain()
print(datetime.datetime.now())
print(cfg)
seed = cfg.seed if cfg.seed != -1 else np.random.randint(0, 10000)
print("seed=", seed)
args = TrainingArguments(
    output_dir=f"../ckpt/pretrain1/exp{cfg.exp}/{os.path.basename(cfg.ckpt)}",
    save_strategy="epoch",
    learning_rate=cfg.lr,
    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=cfg.batch_size,
    num_train_epochs=cfg.epochs,
    weight_decay=cfg.weight_decay,
    warmup_ratio=0.2,
    fp16=True,
    report_to='none',
    dataloader_num_workers=4,
    group_by_length=True,
    save_total_limit=1,
    seed=seed,
)
model = AutoModelForMaskedLM.from_pretrained(cfg.ckpt)
if cfg.gradient_checkpointing:
    model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(cfg.ckpt)
dataset = PretrainDataset1('../data/feedback-prize-2021/', tokenizer, cfg.max_len)
print("len(tokenizer) =", len(tokenizer))
assert 'CLS_CLAIM' in tokenizer.get_vocab()
assert 'CLS_END' in tokenizer.get_vocab()
if cfg.resize_embedding:
    model.resize_token_embeddings(len(tokenizer))
print(f"n_train={len(dataset)}")
trainer = Trainer(
    model,
    args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForWholeWordMask(
        tokenizer=tokenizer, mlm=True, mlm_probability=cfg.mlm_prob),
)
trainer.train()
torch.save(model.state_dict(), f"../ckpt/pretrain1/exp{cfg.exp}/pretrained_model.pt")
save_json({**vars(cfg), 'seed_used': seed}, f"../ckpt/pretrain1/exp{cfg.exp}/config.json")