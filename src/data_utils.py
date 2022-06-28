import os

from tqdm import tqdm

LABEL_MAPPING = {"Ineffective": 0, "Adequate": 1, "Effective": 2}


def prepare_training_data(indir, tokenizer, df):
    training_samples = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        idx = row["essay_id"]
        discourse_text = row["discourse_text"]
        discourse_type = row["discourse_type"]

        filename = os.path.join(indir, idx + ".txt")

        with open(filename, "r") as f:
            text = f.read()

        encoded_text = tokenizer.encode_plus(
            discourse_type + " " + discourse_text,
            text,
            add_special_tokens=False,
        )

        sample = {
            "discourse_id": row["discourse_id"],
            "fold": row["kfold"],
            "input_ids": encoded_text["input_ids"],
            # "discourse_text": discourse_text,
            # "essay_text": text,
            # "mask": encoded_text["attention_mask"],
        }

        if "token_type_ids" in encoded_text:
            sample["token_type_ids"] = encoded_text["token_type_ids"]

        if "discourse_effectiveness" in row:
            label = row["discourse_effectiveness"]
            sample["label"] = LABEL_MAPPING[label]

        training_samples.append(sample)
    return training_samples
