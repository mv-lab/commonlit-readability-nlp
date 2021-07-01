import os
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids

from max.src.fit_bert import fit, NLPModel, Config


class MixupNLPClassificationModel(NLPModel):

    def get_model(self):
        model_config = AutoConfig.from_pretrained(self.config.model_name)
        model_config.num_labels = 1
        model_config.return_dict = True
        return AutoModelForSequenceClassification.from_pretrained(self.config.model_name, config=model_config)

    def permute_batch(self, input_dict):
        shuffle_strength = np.random.beta(0.15, 0.15, 1)[0]

        y = input_dict[self.config.target_column_name]

        batch_permutation = torch.randperm(len(y))
        input_dict[self.config.target_column_name] = (1 - shuffle_strength) * y + \
                                                     shuffle_strength * y[batch_permutation]
        return input_dict, shuffle_strength, batch_permutation

    def forward(self, input_dict):
        if self.trainer.training and random.random() > 0.7:
            input_dict, shuffle_strength, batch_permutation = self.permute_batch(input_dict)
            embeddings = self.model.roberta.embeddings.word_embeddings(input_dict['input_ids'].to(self.device))
            embeddings = (1 - shuffle_strength) * embeddings + shuffle_strength * embeddings[batch_permutation]

            padding_idx = self.model.roberta.embeddings.padding_idx
            position_ids = create_position_ids_from_input_ids(input_ids=input_dict['input_ids'].to(self.device),
                                                              padding_idx=padding_idx).to(self.device)

            logits = self.model(inputs_embeds=embeddings,
                                position_ids=position_ids,
                                attention_mask=input_dict['attention_mask'].to(self.device)
                                ).logits
        else:
            logits = self.model(input_ids=input_dict['input_ids'].to(self.device),
                                attention_mask=input_dict['attention_mask'].to(self.device)
                                ).logits

        input_dict['logits'] = logits
        return input_dict


if __name__ == '__main__':
    df_train = pd.read_csv('../../input/train_folds.csv')
    df_test = pd.read_csv('../../input/test.csv')

    config = Config(root_dir='../../lightning_logs/mixup')
    pl.seed_everything(seed=config.seed)

    return_dict = fit(config=config,
                      df_train=df_train,
                      df_test=df_test,
                      model_class=MixupNLPClassificationModel)
    loss = return_dict['loss']
    print(return_dict['loss'])

    df_oof = return_dict['df_oof']
    df_test_preds = return_dict['df_test_preds']

    experiment_name = config.to_str() + f'oof_loss:_{loss}'

    df_oof.to_csv(os.path.join(config.root_dir, 'df_oof_' + experiment_name + '.csv'), index=False)
    df_test_preds.to_csv(os.path.join(config.root_dir, 'df_test_preds_' + experiment_name + '.csv'), index=False)
