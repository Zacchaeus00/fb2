import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import log_loss
from tqdm import tqdm


def eval_token_cls_model(model, samples, device="cuda"):
    model = model.to(device)
    model.eval()
    predictions = []
    labels = []
    for sample in tqdm(samples):
        input = torch.tensor(sample['input_ids']).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input).logits.squeeze()
        label_idxs = torch.tensor(sample['label_positions'])
        prediction = logits[label_idxs].cpu().detach().numpy()
        predictions.append(prediction)
        labels += sample['raw_labels']
    predictions = np.vstack(predictions)
    probs = softmax(predictions, axis=1)
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    return log_loss(labels, probs, labels=[0, 1, 2])
