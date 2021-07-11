# %% [raw]
# 

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-25T16:15:02.705454Z","iopub.execute_input":"2021-06-25T16:15:02.706225Z","iopub.status.idle":"2021-06-25T16:15:05.000248Z","shell.execute_reply.started":"2021-06-25T16:15:02.706122Z","shell.execute_reply":"2021-06-25T16:15:04.999413Z"}}
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 23:22:27 2019

@author: trushk
"""


import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

import argparse
import os
import random
import pickle
from collections import OrderedDict, defaultdict

ID_COL  = 'id'
TARGET_COL = 'target'
TEXT_COL = 'excerpt'
DEVICE = torch.device('cuda')

preds = 0
num_folds = 5
random_state = 1234


#Create models dir in folder

class BERTDataset(Dataset):
    def __init__(self, review, model_name, target=None, is_test=False):
        self.review = review
        self.target = target
        self.is_test = is_test
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.max_len = args.max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, idx):
        review = str(self.review[idx])
        if args.lower:
            review = review.lower()
        #review = review.replace('\n', '')
        review = ' '.join(review.split())
        global inputs

        inputs = self.tokenizer.encode_plus(
            text=review,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=True
        )
 
        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)

        if self.is_test:
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
            }
        else:
            targets = torch.tensor(self.target[idx], dtype=torch.float)
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
                'targets': targets,
            }

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-25T16:15:05.001808Z","iopub.execute_input":"2021-06-25T16:15:05.002143Z","iopub.status.idle":"2021-06-25T16:15:05.030357Z","shell.execute_reply.started":"2021-06-25T16:15:05.002109Z","shell.execute_reply":"2021-06-25T16:15:05.028923Z"}}
class BERTModel(nn.Module):
    def __init__(self, model_name):
        super(BERTModel, self).__init__()
        self.config = transformers.AutoConfig.from_pretrained(model_name) #, output_hidden_states=True)
        self.bert = transformers.AutoModel.from_pretrained(model_name , output_hidden_states=True)
        self.model_name = model_name
        self.rd_feature_len = 0
        
        if 'distil'  in model_name:
            self.layer_norm = nn.LayerNorm(args.hidden_size)
            
        if args.use_dropout:
            if args.multisample_dropout:
                self.dropouts = nn.ModuleList([
                 nn.Dropout(args.use_dropout) for _ in range(5)
                ])
            else:
                self.dropouts = nn.ModuleList([nn.Dropout(args.use_dropout)])

        # Custom head
        if not args.use_single_fc:
            self.whole_head = nn.Sequential(OrderedDict([
            ('dropout0', nn.Dropout(args.use_dropout)),
            ('l1', nn.Linear(args.fc_size + self.rd_feature_len, 256)),
            ('act1', nn.GELU()),
            ('dropout1', nn.Dropout(args.use_dropout)),
            ('l2', nn.Linear(256, 1))
        ]))
        else:
            self.fc = nn.Linear(args.fc_size + self.rd_feature_len, 1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, ids, mask, rd_features=None, token_type_ids=None):
        # Returns keys(['last_hidden_state', 'pooler_output', 'hidden_states'])
        if token_type_ids is not None:
            output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=True)
        else:
            output = self.bert(ids, attention_mask=mask, return_dict=True)

        #output = self.bert(ids, return_dict=True)

        # Hidden layer
        if args.use_hidden:
          if args.use_hidden == 'last':
              # Last  hidden states
              output = output['hidden_states'][-1]
              output = output.mean(1)
              if args.use_rd_features:
                  output = torch.cat((output, rd_features),1)
                  output = self.layer_norm(output)

          elif args.use_hidden == 'mean_max':
              output = output['last_hidden_state']
              average_pool = torch.mean(output, 1)
              max_pool, _ = torch.max(output, 1)
              output = torch.cat((average_pool, max_pool), 1)
              if args.use_rd_features:
                  output = torch.cat((output, rd_features),1)
                  output = self.layer_norm(output)

          elif args.use_hidden == 'mean':
              hs = output['hidden_states']
              seq_output = torch.cat([hs[-1],hs[-2],hs[-3], hs[-4]], dim=-1)
              input_mask_expanded = mask.unsqueeze(-1).expand(seq_output.size()).float()
              sum_embeddings = torch.sum(seq_output * input_mask_expanded, 1)
              sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
              output = sum_embeddings / sum_mask
              if args.use_rd_features:
                  output = torch.cat((output, rd_features),1)
                  output = self.layer_norm(output)
                
          elif args.use_hidden == 'mean_new':
              hs = output['hidden_states']
              seq_output = torch.cat([hs[-1],hs[-2],hs[-3], hs[-4]], dim=-1)
              seq_max = torch.amax(seq_output, dim=1) 
              input_mask_expanded = mask.unsqueeze(-1).expand(seq_output.size()).float()
              sum_embeddings = torch.sum(seq_output * input_mask_expanded, 1)
              sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
              output = sum_embeddings / sum_mask
              output = torch.cat([output, seq_max], dim=1)
              if args.use_rd_features:
                  output = torch.cat((output, rd_features),1)
                  output = self.layer_norm(output)

        # Pooler
        elif args.use_pooler:
          output = output['pooler_output']
          if args.use_rd_features:
              output = torch.cat((output, rd_features),1)
              output = self.layer_norm(output)
        # Mean of last layer
        elif args.use_last_mean:
          output = output['last_hidden_state']
          input_mask_expanded = mask.unsqueeze(-1).expand(output.size()).float()
          sum_embeddings = torch.sum(output * input_mask_expanded, 1)
          sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
          output = sum_embeddings / sum_mask
          if args.use_rd_features:
              output = torch.cat((output, rd_features),1)
              output = self.layer_norm(output)
        # CLS
        else:
          # Last layer
          output = output['last_hidden_state']
          # CLS token
          output = output[:,0,:]
          if args.use_rd_features:
              output = torch.cat((output, rd_features),1)
              output = self.layer_norm(output)

        #output = self.layer_norm(output)
    
        """
        # Dropout if single FC used
        if args.use_dropout and args.use_single_fc:
          for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.fc(dropout(output))
            else:
                logits += self.fc(dropout(output))
          output = logits/len(self.dropouts)
        elif args.use_single_fc:
            output = self.fc(output)
        """
        
        # Custom head
        if not args.use_single_fc:
            output = self.whole_head(output)
        else:
            output = self.fc(output)
        output = output.squeeze(-1).squeeze(-1)
        return output

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-25T16:15:05.032618Z","iopub.execute_input":"2021-06-25T16:15:05.033084Z","iopub.status.idle":"2021-06-25T16:15:05.043582Z","shell.execute_reply.started":"2021-06-25T16:15:05.032984Z","shell.execute_reply":"2021-06-25T16:15:05.042703Z"}}


def get_bert_predictions(test_data, model_name, model_path):
        print('Getting BERT Embeddings')
        """
        This function validates the model for one epoch through all batches of the valid dataset
        It also returns the validation Root mean squared error for assesing model performance.
        """
        BertModel = BERTModel(model_name=model_name)
        #print(BertModel) 
        BertModel.to(DEVICE) 
        BertModel.load_state_dict(torch.load(model_path), strict=True)

        test_set = BERTDataset(
            review = test_data[TEXT_COL].values,
            target = None,
            model_name = model_name,
            is_test = True

        )

        test_data_loader = DataLoader(
            test_set,
            batch_size = Config.VALID_BS,
            shuffle = False,
            num_workers=8
        )

        prog_bar = tqdm(enumerate(test_data_loader), total=len(test_data_loader))
        BertModel.eval()
        all_predictions = []
        with torch.no_grad():
            for idx, inputs in prog_bar:
                ids = inputs['ids'].to(DEVICE, dtype=torch.long)
                mask = inputs['mask'].to(DEVICE, dtype=torch.long)
                ttis = inputs['token_type_ids'].to(DEVICE, dtype=torch.long)
                if 'distil' in model_name or 'bart' in model_name:
                    ttis = None
                outputs = BertModel(ids=ids, mask=mask, token_type_ids=ttis)
                all_predictions.extend(outputs.cpu().detach().numpy())

        return all_predictions

# %% [markdown]
# 

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-25T16:15:05.045389Z","iopub.execute_input":"2021-06-25T16:15:05.045922Z","iopub.status.idle":"2021-06-25T16:15:05.131116Z","shell.execute_reply.started":"2021-06-25T16:15:05.045885Z","shell.execute_reply":"2021-06-25T16:15:05.1302Z"}}
GET_CV = True
DEBUG = False

if GET_CV:
    df = pd.read_csv('../input/train_folds.csv')
    df = pd.concat([df, df])
    print(df.shape)
    if DEBUG:
        df = df.sample(100)
else:
    df = pd.read_csv('../input/test.csv')
    


class Config:
    seed = 1234
    NB_EPOCHS = 10
    LR = 4e-5
    N_SPLITS = 5
    TRAIN_BS = 32
    VALID_BS = 64
    DBERT_MODELS = ['distilbert', 'xlnet', 't5']
    FILE_NAME = '../input/train.csv'
    scaler = GradScaler()

oof_predictions = np.zeros(len(df))

for fold in range(5):

        print(f"Fold: {fold}")
        print('-'*20)

        # Create fold data
        valid_data = df[df['kfold']==fold].reset_index(drop=True)
        valid_idx = df[df['kfold']==fold].index.values

        pred_df = pd.DataFrame()


        """
        # ## LB 0.471 - electra-large-discriminator
        class BERTModelConfig():
             lower = False
             use_dropout = 0.1
             pretrained_model = False
             use_hidden = False
             hidden_size = 1024
             max_len = 250
             fc_size = 1024
             use_single_fc = False
             use_pooler = False
             use_last_mean = False
             multisample_dropout = False
             use_rd_features = False

        args = BERTModelConfig()


        pred_df[f'electra'] = get_bert_predictions(valid_data, model_name='google/electra-large-discriminator' ,
                                                  model_path=f'../output/electra_l_0630/electra_large_0630/bert_model_fold{fold}.bin'
                                                )

        torch.cuda.empty_cache()
        gc.collect()

        # %% [markdown]
        # ## LB 0.471 - Deberta-large

        # %% [markdown] {"jupyter":{"outputs_hidden":false}}
        #
        class BERTModelConfig():
             lower = False
             use_dropout = 0.1
             pretrained_model = False
             use_hidden = False
             hidden_size = 1024
             max_len = 250
             fc_size = 1024
             use_single_fc = False
             use_pooler = False
             use_last_mean = False
             multisample_dropout = False
             use_rd_features = False

        args = BERTModelConfig()


        pred_df[f'deberta'] = get_bert_predictions(valid_data, model_name='microsoft/deberta-large' ,
                                                  model_path=f'../output/deberta_l_0627/deberta_l_0627/bert_model_fold{fold}.bin'
                                                )

        torch.cuda.empty_cache()
        gc.collect()
   
        """

        # %% [markdown]
        # ## LB 0.465-Roberta-large

        # %% [markdown] {"jupyter":{"outputs_hidden":false}}
        #
        class BERTModelConfig():
             lower = False
             use_dropout = 0.1
             pretrained_model = False
             use_hidden = False
             hidden_size = 1024
             max_len = 250
             fc_size = 1024
             use_single_fc = False
             use_pooler = False
             use_last_mean = False
             multisample_dropout = False
             use_rd_features = False
             use_hidden_4 = False

        args = BERTModelConfig()


        pred_df[f'roberta'] = get_bert_predictions(valid_data, model_name='roberta-large' ,
                                                  model_path=f'../output/roberta_l_0621/roberta_l_0621/bert_model_fold{fold}.bin'
                                                )

        


        
        val_pred_group1_df = pd.DataFrame()
        val_pred_group2_df = pd.DataFrame()
        val_pred_df = pd.DataFrame()

        #val_pred_group1_df['target'] = pred_df[['roberta', 'deberta', 'electra']].mean(axis=1).values.tolist()   
        #val_pred_group2_df['target'] = pred_df[['bart']].mean(axis=1).values.tolist() 
        #val_pred_df['target'] = (val_pred_group1_df['target'] * 0.5) + (val_pred_group2_df['target'] * 0.5)
        
        val_pred_df['target'] = pred_df[['roberta', 'deberta', 'electra']].mean(axis=1).values.tolist()   

        val_pred = val_pred_df['target'].values
        targets = df.iloc[valid_idx,:][TARGET_COL].values
        oof_rmse = np.sqrt(mean_squared_error(targets, val_pred))
        print(f'Fold {fold} RMSE: {oof_rmse}')
        oof_predictions[valid_idx] = val_pred

# %% [code] {"jupyter":{"outputs_hidden":false}}
if GET_CV:
    #predictions = pred_df.mean(axis=1).values.tolist()
    targets = df[TARGET_COL].values
    #rmse = np.sqrt(mean_squared_error(targets, predictions))
    oof_rmse = np.sqrt(mean_squared_error(targets, oof_predictions))
    print(f'RMSE: {oof_rmse}')
else:
    sub_df = pd.read_csv('../input/sample_submission.csv')
    # Mean of all above folds
    sub_df['target'] = pred_df.mean(axis=1).values.tolist()
    print(pred_df.head())
    sub_df.to_csv('submission.csv', index=False)
    print(sub_df.head())
