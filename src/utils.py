import json
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def get_cv(directory):
    scores = []
    for fold in range(5):
        path = os.path.join(directory, f'fold{fold}.json')
        if not os.path.isfile(path):
            return
        with open(path, 'r') as f:
            data = json.load(f)
        scores.append(data['score'])
    assert len(scores) == 5, len(scores)
    cv = np.mean(scores)
    with open(os.path.join(directory, 'cv.txt'), 'w') as f:
        f.write(str(cv))
    print(f"*** CV={cv} is saved to {directory} ***")


def get_oof(directory):
    dfs = []
    for fold in range(5):
        path = os.path.join(directory, f'fold{fold}_oof.csv')
        if not os.path.isfile(path):
            return
        dfs.append(pd.read_csv(path))
    assert len(dfs) == 5, len(dfs)
    df = pd.concat(dfs).reset_index(drop=True)
    path = os.path.join(directory, 'oof.csv')
    df.to_csv(path, index=False)
    for fold in range(5):
        path = os.path.join(directory, f'fold{fold}_oof.csv')
        os.remove(path)
    print(f"*** OOF saved to {path} ***")


def try_train(trainer, sleep=-1):
    try:
        trainer.train()
    except RuntimeError:
        print(f"BAD GPU: sleep {sleep} hours.")
        if sleep != -1:
            time.sleep(3600 * sleep)
        exit(1)


def check_gpu(th=0.1, sleep=5):
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    pct_used = info.used / info.total
    if pct_used > 0.1:
        print("GPU already in use")
        if sleep != -1:
            time.sleep(3600 * sleep)
        exit(1)
    else:
        print("GPU checked")
        return