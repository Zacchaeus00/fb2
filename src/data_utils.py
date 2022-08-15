import os

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from glob import glob

LABEL_MAPPING = {"Ineffective": 0, "Adequate": 1, "Effective": 2}
disc_types = [
    "Claim",
    "Concluding Statement",
    "Counterclaim",
    "Evidence",
    "Lead",
    "Position",
    "Rebuttal",
]
cls_tokens_map = {label: f"[CLS_{label.upper()}]" for label in disc_types}
end_tokens_map = {label: f"[END_{label.upper()}]" for label in disc_types}


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
        self.features = ['input_ids', 'attention_mask', 'label', 'labels']

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


def get_tag_v2(discourse_type, end=False):
    discourse_type = discourse_type.lower()
    if end:
        return f'<{discourse_type} end>'
    return f'<{discourse_type} start>'


def fix_sidx(sidx, text):
    while sidx > 0 and text[sidx].isalpha() and text[sidx - 1].isalpha():
        # print(f"{text[sidx:sidx+100]} -> {text[sidx-1:sidx-1+100]}")
        sidx -= 1
    return sidx


def insert_tag(text, dtext, dtype, start=0, fix=False):
    tag = get_tag(dtype)
    sidx = text.find(dtext, start)
    if fix:
        sidx = fix_sidx(sidx, text)
    if sidx == -1:
        raise KeyError
    text = text[:sidx] + ' ' + tag + ' ' + text[sidx:]
    eidx = sidx + len(' ' + tag + ' ') + len(dtext)
    return text, sidx, eidx


def insert_tag_v2(text, dtext, dtype, start=0, fix=False):
    stag = get_tag_v2(dtype)
    etag = get_tag_v2(dtype, end=True)
    sidx = text.find(dtext, start)
    if fix:
        sidx = fix_sidx(sidx, text)
    if sidx == -1:
        raise KeyError
    text = text[:sidx] + ' ' + stag + ' ' + text[sidx:]
    eidx = sidx + len(' ' + stag + ' ') + len(dtext)
    text = text[:eidx] + ' ' + etag + ' ' + text[eidx:]
    return text, sidx, eidx


def prepare_samples(essay, train, fix, end=False):
    samples = []
    for eid in tqdm(essay.index):
        text = essay[eid]
        df = train[train['essay_id'] == eid]
        idxs = []
        labels = []
        eidx = 0
        dids = []
        for _, row in df.iterrows():
            dtype = row['discourse_type']
            dtext = row['discourse_text_processed']
            dids.append(row['discourse_id'])
            label = LABEL_MAPPING[row['discourse_effectiveness']]
            if end:
                text, sidx, eidx = insert_tag_v2(text, dtext, dtype, start=eidx, fix=fix)
            else:
                text, sidx, eidx = insert_tag(text, dtext, dtype, start=eidx, fix=fix)
            idxs.append([sidx, eidx])
            labels.append(label)
        assert (idxs == list(sorted(idxs))), idxs
        assert df['kfold'].nunique() == 1, df['kfold'].nunique()
        samples.append(
            {'text': text, 'spans': idxs, 'raw_labels': labels, 'fold': df['kfold'].unique()[0], 'essay_id': eid,
             'discourse_ids': dids})
    return samples


def prepare_samples_with_prompt(essay, train, prompt, sep='[SEP]'):
    samples = []
    for eid in tqdm(essay.index):
        text = essay[eid]
        ptext = prompt[eid]
        df = train[train['essay_id'] == eid]
        idxs = []
        labels = []
        eidx = 0
        dids = []
        for _, row in df.iterrows():
            dtype = row['discourse_type']
            dtext = row['discourse_text_processed']
            dids.append(row['discourse_id'])
            label = LABEL_MAPPING[row['discourse_effectiveness']]
            text, sidx, eidx = insert_tag_v2(text, dtext, dtype, start=eidx, fix=False)
            idxs.append([sidx, eidx])
            labels.append(label)
        assert (idxs == list(sorted(idxs))), idxs
        assert df['kfold'].nunique() == 1, df['kfold'].nunique()
        text = ptext + sep + text
        idxs = [list(map(lambda x: x + len(ptext) + len(sep), span)) for span in idxs]
        samples.append(
            {'text': text, 'spans': idxs, 'raw_labels': labels, 'fold': df['kfold'].unique()[0], 'essay_id': eid,
             'discourse_ids': dids})
    return samples


def tokenize_samples(samples, tokenizer, pooling):
    for sample in tqdm(samples):
        enc = tokenizer(sample['text'], return_offsets_mapping=True, add_special_tokens=False)
        seq_len = len(enc['input_ids'])
        label = [-100 for _ in range(seq_len)]

        # 1. mean
        if pooling == 'mean':
            label_positions = [[] for _ in range(len(sample['spans']))]
            for i in range(seq_len):
                for j, (s, e) in enumerate(sample['spans']):
                    if s <= enc['offset_mapping'][i][0] < e and enc['offset_mapping'][i][1] > enc['offset_mapping'][i][
                        0]:
                        label[i] = sample['raw_labels'][j]
                        label_positions[j].append(i)
                        break

        # 2. cls
        elif pooling == 'cls':
            label_positions = []
            j = 0
            for i in range(seq_len):
                if j == len(sample['raw_labels']):
                    break
                s, e = sample['spans'][j]
                if enc['offset_mapping'][i][0] >= s and e > s:
                    label[i] = sample['raw_labels'][j]
                    j += 1
                    label_positions.append(i)
        else:
            raise NotImplementedError
        sample['label'] = label
        sample['label_positions'] = label_positions
        for k, v in enc.items():
            sample[k] = v
        nlabel_assigned = len([l for l in sample['label'] if l != -100])
        assert len(sample['raw_labels']) == len(sample['label_positions'])
        assert len(sample['spans']) == len(sample['label_positions'])
        if pooling == 'cls':
            assert (nlabel_assigned == len(sample['raw_labels'])), f"{nlabel_assigned}, {len(sample['raw_labels'])}"
    return samples


def prepare_data_token_cls(essay, train, tokenizer, pooling='cls', fix=False, end=False):
    samples = prepare_samples(essay, train, fix, end)
    samples = tokenize_samples(samples, tokenizer, pooling)
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


class Inserter:
    def __init__(self, text):
        self.raw_text = text
        self.text = text
        self.offset = 0

    def insert(self, tag, idx):
        idx += self.offset
        #         assert 0 <= idx and idx <= len(self.text), [idx, len(self.text)]
        idx = max([min([idx, len(self.text)]), 0])
        if idx == len(self.text) or self.text[idx] != ' ':
            tag = tag + ' '
        if idx == 0 or self.text[idx - 1] != ' ':
            tag = ' ' + tag
        self.offset += len(tag)
        self.text = self.text[:idx] + tag + self.text[idx:]

    def get(self):
        return self.text


class PretrainDataset1(torch.utils.data.Dataset):
    def __init__(self, dir2021, tokenizer, max_len):
        train = pd.read_csv(os.path.join(dir2021, 'train.csv'))
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(cls_tokens_map.values()) + list(end_tokens_map.values())}
        )
        self.tokenizer = tokenizer
        self.max_len = max_len
        texts = []
        for eid in tqdm(train.id.unique()):
            text = open(os.path.join(dir2021, f'train/{eid}.txt')).read()
            df = train[train['id'] == eid].reset_index(drop=True)
            inserter = Inserter(text)
            for i, row in df.iterrows():
                inserter.insert(cls_tokens_map[row['discourse_type']], int(row['discourse_start']))
                inserter.insert(end_tokens_map[row['discourse_type']], int(row['discourse_end']))
            texts.append(inserter.get())
        self.texts = texts
        print(len(texts), "samples")
        print(texts[0])
        print("vocab size:", len(tokenizer))

    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx], max_length=self.max_len, truncation=True)

    def __len__(self):
        return len(self.texts)


if __name__ == '__main__':
    import pandas as pd
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    # essay = pd.read_csv('../data/essay_processed.csv')
    # essay = essay.set_index('essay_id').squeeze()
    # train = pd.read_csv('../data/train_processed.csv')
    # samples = prepare_data_token_cls(essay, train, tokenizer, pooling='mean', end=True)
    # print(samples[:5])
    ds = PretrainDataset1('../data/feedback-prize-2021', tokenizer, 1024)
