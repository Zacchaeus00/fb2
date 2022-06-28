import json
import os
import pickle
import random

import numpy as np
import torch


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