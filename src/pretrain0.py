import argparse
import shutil

import torch
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForWholeWordMask

from data_utils import PretrainDataset
from utils import save_json

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
    return parser.parse_args()


cfg = parse_args_pretrain()
args = TrainingArguments(
    output_dir=f"../ckpt/pretrain0/exp{cfg.exp}/tmp",
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
    seed=cfg.seed,
)
model = AutoModelForMaskedLM.from_pretrained(cfg.ckpt)
tokenizer = AutoTokenizer.from_pretrained(cfg.ckpt)
dataset = PretrainDataset('../data/feedback-prize-effectiveness/train', tokenizer, cfg.max_len)
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
torch.save(model.state_dict(), f"../ckpt/pretrain0/exp{cfg.exp}/pretrained_model.pt")
shutil.rmtree(f"../ckpt/pretrain0/exp{cfg.exp}/tmp")
save_json(vars(cfg), f"../ckpt/pretrain0/exp{cfg.exp}/config.json")