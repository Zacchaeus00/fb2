import os

from tqdm import tqdm

LABEL_MAPPING = {"Ineffective": 0, "Adequate": 1, "Effective": 2}


def prepare_training_data(indir, tokenizer, df, max_len):
    training_samples = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        idx = row["essay_id"]
        discourse_text = row["discourse_text"]
        discourse_type = row["discourse_type"]

        filename = os.path.join(indir, idx + ".txt")

        with open(filename, "r") as f:
            text = f.read()

        encoding = tokenizer.encode_plus(
            discourse_type + " " + discourse_text,
            text,
            truncation=True,
            max_length=max_len
        )

        sample = {
            "discourse_id": row["discourse_id"],
            "fold": row["kfold"],
            **encoding,
        }

        if "discourse_effectiveness" in row:
            label = row["discourse_effectiveness"]
            sample["label"] = LABEL_MAPPING[label]

        training_samples.append(sample)
    return training_samples


class FB2Dataset:
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_ids": sample['input_ids'],
            "attention_mask": sample['attention_mask'],
            "label": sample['label'],
        }