import argparse

from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForWholeWordMask

from data_utils import PretrainDataset


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
    return parser.parse_args()


cfg = parse_args_pretrain()
args = TrainingArguments(
    output_dir=f"../ckpt/pretrain0/exp{cfg.exp}",
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
model = AutoModelForMaskedLM.from_pretrained(cfg.pretrained_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_checkpoint)
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
