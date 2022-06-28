import os

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

LABEL_MAPPING = {"Ineffective": 0, "Adequate": 1, "Effective": 2}


def prepare_data(indir, tokenizer, df, max_len):
    training_samples = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        id_ = row["essay_id"]
        discourse_text = row["discourse_text"]
        discourse_type = row["discourse_type"]

        filename = os.path.join(indir, id_ + ".txt")

        with open(filename, "r") as f:
            text = f.read()

        encoding = tokenizer.encode_plus(
            discourse_type + " " + discourse_text,
            text,
            truncation='only_second',
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


def prepare_data_mp(indir, tokenizer, df, max_len, j=8):
    training_samples = []

    df_splits = np.array_split(df, j)

    results = Parallel(n_jobs=j, backend="multiprocessing")(
        delayed(prepare_data)(indir, tokenizer, df, max_len) for df in df_splits
    )
    for result in results:
        training_samples.extend(result)

    return training_samples


class FB2Dataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.features = ['input_ids', 'attention_mask', 'token_type_ids', 'label']

    def __getitem__(self, i):
        return {k: v for k, v in self.samples[i].items() if k in self.features}

    def __len__(self):
        return len(self.samples)
