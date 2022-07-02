import os

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from glob import glob

LABEL_MAPPING = {"Ineffective": 0, "Adequate": 1, "Effective": 2}


def prepare_data(indir, tokenizer, df, max_len):
    training_samples = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        id_ = row["essay_id"]
        discourse_text = row["discourse_text"].lower()
        discourse_type = row["discourse_type"].lower()

        filename = os.path.join(indir, id_ + ".txt")

        with open(filename, "r") as f:
            text = f.read().lower()

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
        data = {k: v for k, v in self.samples[i].items() if k in self.features}
        if 'label' in data:
            data['labels'] = data['label']
            del data['label']
        return data

    def __len__(self):
        return len(self.samples)


def get_tag(discourse_type):
    return f'<{discourse_type.lower()}>'


def insert_tag(text, dtext, dtype, start=0):
    tag = get_tag(dtype)
    sidx = text.find(dtext, start)
    if sidx == -1:
        raise KeyError
    text = text[:sidx] + ' ' + tag + ' ' + text[sidx:]
    eidx = sidx + len(' ' + tag + ' ') + len(dtext)
    return text, sidx, eidx


def prepare_data_token_cls(essay, train, tokenizer):
    samples = []
    for eid in tqdm(essay.index):
        text = essay[eid]
        df = train[train['essay_id'] == eid]
        idxs = []
        labels = []
        eidx = 0
        for _, row in df.iterrows():
            dtype = row['discourse_type']
            dtext = row['discourse_text_processed']
            did = row['discourse_id']
            label = LABEL_MAPPING[row['discourse_effectiveness']]
            text, sidx, eidx = insert_tag(text, dtext, dtype, start=eidx)
            idxs.append([sidx, eidx])
            labels.append(label)
        assert (idxs == list(sorted(idxs))), idxs
        assert df['kfold'].nunique() == 1, df['kfold'].nunique()
        samples.append({'text': text, 'spans': idxs, 'raw_labels': labels, 'fold': df['kfold'].unique()[0], 'essay_id': eid, 'discourse_id': did})
    for sample in tqdm(samples):
        enc = tokenizer(sample['text'], return_offsets_mapping=True, add_special_tokens=False)
        seq_len = len(enc['input_ids'])
        label = [-100 for _ in range(seq_len)]
        # 1. mean
        # for i in range(seq_len):
        #     for j, (s, e) in enumerate(sample['spans']):
        #         if enc['offset_mapping'][i][0] >= s and enc['offset_mapping'][i][0] < e and e > s:
        #             label[i] = sample['raw_labels'][j]
        #             break

        # 2. cls
        j = 0
        label_positions = []
        for i in range(seq_len):
            if j == len(sample['raw_labels']):
                break
            s, e = sample['spans'][j]
            if enc['offset_mapping'][i][0] >= s and e > s:
                label[i] = sample['raw_labels'][j]
                j += 1
                label_positions.append(i)
        sample['label'] = label
        sample['label_positions'] = label_positions
        for k, v in enc.items():
            sample[k] = v
        nlabel_assigned = len([l for l in sample['label'] if l != -100])
        assert (nlabel_assigned == len(sample['raw_labels'])), f"{nlabel_assigned}, {len(sample['raw_labels'])}"
    return samples


class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, indir, tokenizer, max_len):
        paths = glob(os.path.join(indir, '*.txt'))
        self.texts = []
        for p in paths:
            with open(p, 'r') as f:
                self.texts.append(f.read().lower())
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx], add_special_tokens=False, max_length=self.max_len, truncation=True)

    def __len__(self):
        return len(self.texts)


