import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import AutoConfig, AutoModelForSequenceClassification

from max.src.fit_bert import fit, Config, NLPClassificationModel


@dataclass
class EmbeddingConfig(Config):
    a: float = 0.2

    def to_str(self):
        return super().to_str() \
               + f'a:{self.a}_'

def cross_entropy(self, logits1, logits2, target1, target2):
    log_q = torch.nn.functional.log_softmax(torch.stack([logits1, logits2], dim=1))
    p = torch.nn.functional.softmax(torch.stack([target1.float(), target2.float()], dim=1))
    cross_entropy = -(p * log_q).sum() / p.shape[0]
    return cross_entropy


class NLPClassificationModelWIthProbas(NLPClassificationModel):

    def get_model(self):
        model_config = AutoConfig.from_pretrained(self.config.model_name)
        model_config.num_labels = 1
        model_config.return_dict = True
        model_config.output_hidden_states = True
        return AutoModelForSequenceClassification.from_pretrained(self.config.model_name,
                                                                  config=model_config)

    def dict_loss(self, input_dict):
        logits = input_dict['logits']
        y_true = input_dict[self.config.target_column_name].type(torch.float32).to(logits.device)
        regression_loss = self.loss_function(logits, y_true)

        # p_ij[i, j] = torch.exp(y_true[i]) / (torch.exp(y_true[i]) + torch.exp(y_true[j]]))
        p_ij = torch.exp(y_true) / (torch.exp(y_true[:, 0, None]) + torch.exp(y_true[None, :, 0]))
        p_ij_pred = torch.exp(logits) / (torch.exp(logits[:, 0, None]) + torch.exp(logits[None, :, 0]))

        # p_ij_pred_logits[i, j] = logits[j] - logits[i]
        p_ij_pred_logits = logits[None, :, 0] - logits[:, 0, None]
        probability_loss = nn.BCEWithLogitsLoss()(p_ij, p_ij_pred_logits)

        return regression_loss + self.config.a * probability_loss


if __name__ == '__main__':
    df_train = pd.read_csv('train_folds.csv')
    df_test = pd.read_csv('../../input/test.csv')

    config = EmbeddingConfig(root_dir='../../lightning_logs/probability_loss',
                             optimizer_name='AdamWDifferential',
                             lr=1e-5,
                             scheduler='plateau',
                             epochs=5,
                             batch_size=16,
                             a=0.1,
                             )

    pl.seed_everything(seed=config.seed)

    overwrite_train_params = dict(val_check_interval=0.5,
                                  accumulate_grad_batches=1)
    # 0.5236278885115938 with b = 0.1
    return_dict = fit(config=config,
                      df_train=df_train,
                      df_test=df_test,
                      model_class=NLPClassificationModelWIthProbas,
                      overwrite_train_params=overwrite_train_params)
    loss = return_dict['loss']
    print(return_dict['loss'])

    df_oof = return_dict['df_oof']
    df_test_preds = return_dict['df_test_preds']

    experiment_name = config.to_str() + f'oof_loss:_{loss}'

    df_oof.to_csv(os.path.join(config.root_dir, 'df_oof_' + experiment_name + '.csv'), index=False)
    df_test_preds.to_csv(os.path.join(config.root_dir, 'df_test_preds_' + experiment_name + '.csv'), index=False)
