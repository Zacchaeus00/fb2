import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import log_loss
from tqdm import tqdm


def get_score(logits, labels):
    probs = softmax(logits, axis=1)
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    return log_loss(labels, probs, labels=[0, 1, 2])


def get_score_pt(logits, labels):
    logits = torch.tensor(logits, requires_grad=False)
    labels = torch.tensor(labels, requires_grad=False)
    return torch.nn.functional.cross_entropy(logits, labels).item()


def eval_token_cls_model(model, samples, device="cuda"):
    model = model.to(device)
    model.eval()
    predictions = []
    labels = []
    for sample in tqdm(samples):
        input = torch.tensor(sample['input_ids']).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(model, 'logits'):
                logits = model(input).logits.squeeze()
            else:
                logits = model(input)[0].squeeze()
        label_idxs = torch.tensor(sample['label_positions'])
        prediction = logits[label_idxs].cpu().detach().numpy()
        predictions.append(prediction)
        labels += sample['raw_labels']
    predictions = np.vstack(predictions)
    dids = []
    for s in samples:
        dids += s['discourse_ids']
    assert (len(dids) == predictions.shape[0]), [len(dids), predictions.shape[0]]
    oof_df = pd.DataFrame({'discourse_id': dids,
                           'logits': [predictions[i] for i in range(len(dids))]})
    return get_score(predictions, labels), oof_df
