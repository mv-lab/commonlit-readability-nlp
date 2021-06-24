import gc
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import torch
from torch import TensorDataset, DataLoader
from transformers import AutoModelForSequenceClassification

from tqdm import tqdm

from .utils import add_center_to_folds, auto_tokenize

device = 'cuda'

ROOT_PATH = '/content/drive/MyDrive/commonlit'
DATA_PATH = ROOT_PATH + '/data'
FOLD_PATH = DATA_PATH + '/los_folds.csv'


def get_model_preds(model_paths, input_ids, attention_masks):
    dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(dataset, batch_size=128, pin_memory=True)

    preds_list = []

    for fold in range(len(model_paths)):

        model = AutoModelForSequenceClassification.from_pretrained(
            model_paths[fold],
            num_labels=1,
            output_attentions=False,
            output_hidden_states=False,
        ).to(device)
        model.eval()

        preds = []

        for batch in tqdm(dataloader):
            with torch.no_grad():
                output = model(batch[0].to(device), batch[1].to(device))
                output = output.logits.detach().cpu().numpy().ravel().tolist()
                preds.extend(output)

        del model
        torch.cuda.empty_cache()
        gc.collect()

        preds_list.append(preds)

    return np.array(preds_list)


def get_valid_mse(fold, model, model_type='roberta-large'):
    data = pd.read_csv(FOLD_PATH)
    data = add_center_to_folds(data)
    data = data[data.kfold == fold]

    inputs, attn_msks, _ = auto_tokenize(data.excerpt, data.target.to_list(), model_type)
    preds = get_model_preds([model], inputs, attn_msks)

    return mean_squared_error(preds[0], data.target, squared=False)