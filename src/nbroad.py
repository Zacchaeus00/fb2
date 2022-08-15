import codecs
import os
import re
from functools import partial
from pathlib import Path
from typing import Tuple

import pandas as pd
from datasets import Dataset
from text_unidecode import unidecode
from transformers import AutoTokenizer
LABEL_MAPPING = {"Ineffective": 0, "Adequate": 1, "Effective": 2}

def fix_florida(train):
    train.loc[13906, 'discourse_text'] = train.loc[13906, 'discourse_text'].replace('florida', 'LOCATION_NAME')
    return train

# https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313330
def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end

def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end

def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def read_text_files(example, data_dir):
    id_ = example["essay_id"]
    with open(data_dir / "train" / f"{id_}.txt", "r") as fp:
        example["text"] = resolve_encodings_and_normalize(fp.read())
    return example

def get_dataset(cfg):
    # Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
    codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
    codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

    data_dir = Path(cfg["data_dir"])
    train_df = pd.read_csv(data_dir / "train.csv")
    train_df = fix_florida(train_df)

    text_ds = Dataset.from_dict({"essay_id": train_df.essay_id.unique()})

    text_ds = text_ds.map(
        partial(read_text_files, data_dir=data_dir),
        num_proc=cfg["num_proc"],
        batched=False,
        desc="Loading text files",
    )

    text_df = text_ds.to_pandas()

    train_df["discourse_text"] = [
        resolve_encodings_and_normalize(x) for x in train_df["discourse_text"]
    ]
    train_df = train_df.merge(text_df, on="essay_id", how="left")

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

    # label2id = {
    #     "Adequate": 0,
    #     "Effective": 1,
    #     "Ineffective": 2,
    # }
    label2id = LABEL_MAPPING

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(cls_tokens_map.values()) + list(end_tokens_map.values())}
    )
    cls_id_map = {
        label: tokenizer.encode(tkn)[1]
        for label, tkn in cls_tokens_map.items()
    }

    def find_positions(example):

        text = example["text"][0]

        # keeps track of what has already
        # been located
        min_idx = 0

        # stores start and end indexes of discourse_texts
        idxs = []

        for dt in example["discourse_text"]:
            # calling strip is essential
            matches = list(re.finditer(re.escape(dt.strip()), text))

            # If there are multiple matches, take the first one
            # that is past the previous discourse texts.
            if len(matches) > 1:
                for m in matches:
                    if m.start() >= min_idx:
                        break
            # If no matches are found
            elif len(matches) == 0:
                idxs.append([-1])  # will filter out later
                continue
                # If one match is found
            else:
                m = matches[0]

            idxs.append([m.start(), m.end()])

            min_idx = m.start()

        return idxs

    def tokenize(example):
        example["idxs"] = find_positions(example)

        text = example["text"][0]
        chunks = []
        labels = []
        prev = 0

        zipped = zip(
            example["idxs"],
            example["discourse_type"],
            example["discourse_effectiveness"],
        )
        for idxs, disc_type, disc_effect in zipped:
            # when the discourse_text wasn't found
            if idxs == [-1]:
                continue

            s, e = idxs

            # if the start of the current discourse_text is not
            # at the end of the previous one.
            # (text in between discourse_texts)
            if s != prev:
                chunks.append(text[prev:s])
                prev = s

            # if the start of the current discourse_text is
            # the same as the end of the previous discourse_text
            if s == prev:
                chunks.append(cls_tokens_map[disc_type])
                chunks.append(text[s:e])
                chunks.append(end_tokens_map[disc_type])

            prev = e

            labels.append(label2id[disc_effect])

        tokenized = tokenizer(
            " ".join(chunks),
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        tokenized['raw_labels'] = labels

        # at this point, labels is not the same shape as input_ids.
        # The following loop will add -100 so that the loss function
        # ignores all tokens except CLS tokens

        # idx for labels list
        idx = 0
        final_labels = []
        label_positions = []
        for lp, id_ in enumerate(tokenized["input_ids"]):
            # if this id belongs to a CLS token
            if id_ in cls_id_map.values():
                label_positions.append(lp)
                final_labels.append(labels[idx])
                idx += 1
            else:
                # -100 will be ignored by loss function
                final_labels.append(-100)

        tokenized["label_positions"] = label_positions
        tokenized["labels"] = final_labels

        return tokenized

    # make lists of discourse_text, discourse_effectiveness
    # for each essay
    grouped = train_df.groupby(["essay_id"]).agg(list)

    ds = Dataset.from_pandas(grouped)

    ds = ds.map(
        tokenize,
        batched=False,
        num_proc=cfg["num_proc"],
        desc="Tokenizing",
    )
    return ds, tokenizer

if __name__ == '__main__':
    cfg = {
        'num_proc': 2,
        "data_dir": "../data/feedback-prize-effectiveness",
        "model_name_or_path": "microsoft/deberta-v3-base",
    }
    ds, tokenizer = get_dataset(cfg)
    # print(ds)
    # print(ds[0])
    for sample in ds:
        if len(sample['discourse_id']) != len(sample['label_positions']):
            print(sample['text'])
    # print(tokenizer.decode(ds[0]['input_ids']))
    # fold_df = pd.read_csv('../data/train_folds.csv')
    # folds = []
    # for sample in ds:
    #     eid = sample['essay_id']
    #     df = fold_df[fold_df['essay_id']==eid]
    #     assert(df['kfold'].nunique()==1)
    #     folds.append(df['kfold'].values[0])
    # ds = ds.add_column("fold", folds)
    # print(ds)
