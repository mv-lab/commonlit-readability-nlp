#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 23:22:27 2019

@author: trushk
"""


import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from kaggler.model import AutoLGB

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime as dt
import os
import json
import gc
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

import argparse
import os
import random
import textstat
import pickle
import readability
import syntok.segmenter as segmenter
from laserembeddings import Laser
from sklearn.decomposition import PCA
from collections import OrderedDict, defaultdict


parser = argparse.ArgumentParser(description='Process pytorch params.')

parser.add_argument('-algo', type=str, default='lgb', help='Algorithm to use')
parser.add_argument('-use_autolgb', action='store_true', help='Algorithm to use')
parser.add_argument('-model_dir', type=str, default='test', help='output dir to store results')
parser.add_argument('-batch_size', type=int, default=32, help='batch_size')
parser.add_argument('-get_readability_features', action='store_true', help='Add bert features')
parser.add_argument('-get_bert', action='store_true', help='Add bert features')
parser.add_argument('-get_tfidf', action='store_true',help='Add tfidf features')
parser.add_argument('-get_laser', action='store_true',help='Add laser features')
parser.add_argument('-gpu', type=int, default=[0,1], nargs='+', help='use gpu')
parser.add_argument('-use_pca', type=int,help='PCA')
parser.add_argument('-debug', action='store_true',help='debug mode. small batch size')
parser.add_argument('-use_hidden', action='store_true', help='Use BERT hidden layers vs CLS token')
parser.add_argument('-use_dropout', action='store_true', help='Use dropout layer')
parser.add_argument('-lower', action='store_true', help='Use dropout layer')
parser.add_argument('-device', type=str, default='cuda', help='Use dropout layer')


args = parser.parse_args()


if args.get_laser:
   laser = Laser()


os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)

ID_COL  = 'id'
TARGET_COL = 'target'
TEXT_COL = 'excerpt'
DEVICE = torch.device(args.device)

class Config:
    seed = 1234
    NB_EPOCHS = 10
    LR = 4e-5
    N_SPLITS = 5
    TRAIN_BS = 32
    VALID_BS = 64
    DBERT_MODELS = ['distilbert-base-uncased', 'xlnet', 't5']
    FILE_NAME = '../input/train.csv'
    scaler = GradScaler()
    MAX_LEN = 256

#Extract categorical columns for model
def test():
  for col in X.columns:
    
    if X[col].dtype.name == 'object':
        cat_cols.append(col)
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')
        
    elif X[col].dtype.name == 'category' : 
        cat_cols.append(col)
        
preds = 0
num_folds = 5
random_state = 1234
val_aucs = []


#Create models dir in folder
model_folder = '../output'
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

#Create dir with lgb name + timestamp
#timestamp = '{:%d%b%y_%H%M%S}'.format(dt.now())
op_folder = model_folder+'/'+args.model_dir+'/'
if not os.path.exists(op_folder):
    os.mkdir(op_folder)




gc.collect()

class BERTDataset(Dataset):
    def __init__(self, review, model_name, target=None, is_test=False):
        self.review = review
        self.target = target
        self.is_test = is_test
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.max_len = Config.MAX_LEN

    def __len__(self):
        return len(self.review)

    def __getitem__(self, idx):
        review = str(self.review[idx])
        if args.lower:
           review = review.lower()
        review = review.replace('\n', '')
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


class BERTModel(nn.Module):
    def __init__(self, model_name, multisample_dropout=True):
        super(BERTModel, self).__init__()
        self.model_name = model_name
        if args.use_rd_features:
            self.rd_feature_len = 21
            if args.use_hidden == 'mean':
                self.layer_norm = nn.LayerNorm(args.hidden_size*3 + self.rd_feature_len)
            else:
                self.layer_norm = nn.LayerNorm(args.hidden_size + self.rd_feature_len)
        else:
            self.rd_feature_len = 0
        self.config = transformers.AutoConfig.from_pretrained(model_name)
        if 't5' in model_name:
            self.bert = transformers.T5EncoderModel.from_pretrained(model_name)
            self.pooler = T5Pooler(args.hidden_size)
        else:
            self.bert = transformers.AutoModel.from_pretrained(model_name, output_hidden_states=True)

        if args.use_dropout:
           if multisample_dropout:
               self.dropouts = nn.ModuleList([
                 nn.Dropout(args.use_dropout) for _ in range(5)
               ])
           else:
              self.dropouts = nn.ModuleList([nn.Dropout(args.use_dropout)])

        # Custom head
        if not args.use_single_fc:
            self.whole_head = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(args.use_dropout)),
            ('l1', nn.Linear(args.fc_size + self.rd_feature_len, 256)),
            ('act1', nn.GELU()),
            ('dropout', nn.Dropout(args.use_dropout)),
            ('l2', nn.Linear(256, 1))
        ]))
        else:
            self.fc = nn.Linear(args.hidden_size + self.rd_feature_len, 1)
        #self.fc = nn.Conv1d(args.hidden_size, 1, kernel_size=3) 
        #self._init_weights(self.fc)

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
              seq_output = torch.cat([hs[-1],hs[-2],hs[-3]], dim=-1)
              input_mask_expanded = mask.unsqueeze(-1).expand(seq_output.size()).float()
              sum_embeddings = torch.sum(seq_output * input_mask_expanded, 1)
              sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
              output = sum_embeddings / sum_mask
              if args.use_rd_features:
                  output = torch.cat((avg_output, rd_features),1)
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
        # Custom head
        if not args.use_single_fc:
            if args.use_custom_head:
                output = self.custom_head(output)
            else:
                output = self.whole_head(output)

        output = output.squeeze(-1).squeeze(-1)
        return output




def get_laser_embeddings(df):
   print('Getting Laser Embeddings')
   sentences = df[TEXT_COL]
   embeddings = laser.embed_sentences(sentences,lang='en')
   return embeddings



def get_bert_predictions(test_data, model_name, model_path):
        print('Getting BERT Embeddings')
        """
        This function validates the model for one epoch through all batches of the valid dataset
        It also returns the validation Root mean squared error for assesing model performance.
        """
        BertModel = BERTModel(model_name=model_name)
        #print(BertModel) 
        BertModel.to(DEVICE)
        if args.device == 'cuda':
            BertModel.load_state_dict(torch.load(model_path), strict=True)
        else:
            BertModel.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')), strict=True)

        test_set = BERTDataset(
            review = test_data[TEXT_COL].values,
            target = test_data[TARGET_COL].values,
            model_name = model_name
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
                if model_name in Config.DBERT_MODELS:
                   outputs = BertModel(ids=ids, mask=mask)
                else:
                   outputs = BertModel(ids=ids, mask=mask, token_type_ids=ttis)
                all_predictions.extend(outputs.cpu().detach().numpy())

        return np.array(all_predictions)


if args.algo == 'xgb':
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'max_depth': 8,
        'colsample_bytree': 0.7,
        'subsample': 0.8,
        'gamma': 1,
        }
elif args.algo == 'lgb':
    params = {
          'metric': 'rmse',
          'boosting_type': 'gbdt',
          'n_estimators': 10000,
          'learning_rate': 0.01,
          'max_depth': -1,    #-1 is no limit. deeper trees can lead to overfitting
          #'num_leaves': 128, #128  #increase for better accuracy but can lead to overfitting
          #'min_data_in_leaf': 100, #100 #minimum number of samples to group in leaf. increase to prevent overfitting
          'colsample_bytree': 0.7, #0.4 #column sampling fraction
          'subsample_freq': 5,   #row sampling freq
          'subsample': 0.9,  #0.5   #row sampling fraction
          'bagging_seed': 13,
         # 'bagging_freq':10
          'verbosity': -1,
       #   'reg_alpha': 0.3, #0.3   #regularization params
       #   'reg_lambda': 0.6, #0.6 
          'random_state': 47,
          'num_threads': 12,  #num cores
      #    'early_stopping_round': 3000
         }
else:
    params = None

def get_3k_feat(text):
    with open('./3k_words.pkl', 'rb') as f:
        words_3k = pickle.load(f)
    count = 0
    for word in text.split(' '):
        if word not in words_3k:
            count += 1
    return count


def get_tfidf_features(df, max_features):
    sentences = df[TEXT_COL]
    #sentences = list(df['title'].map(preprocess_tfidf))
    model = TfidfVectorizer(stop_words = 'english', ngram_range=(1,1),  binary = True, max_features = max_features,
                            strip_accents='unicode',
                            lowercase=True,
                            analyzer='char_wb',
                            #token_pattern=r"(?u)\b\w+\b",  
                            #max_df = 0.5,
                            norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False,
                            )
    text_embeddings = model.fit_transform(sentences).toarray()
    return text_embeddings

def get_textstat_readability_features(df):
   print(f'Getting readability features')
   read_features = ['flesch_reading_ease', 'avg_sentence_length', 'avg_syllables_per_word', 'syllable_count', 'avg_character_per_word', 'polysyllabcount', 'automated_readability_index',
                    'dale_chall_readability_score', 'linsear_write_formula', 'smog_index', 'gunning_fog', 'coleman_liau_index']

   features = []
   for feat in read_features:
       #print(f'Getting readability feature: {feat}')
       method = getattr(textstat, feat)
       df[feat] = df[TEXT_COL].apply(method)

   return df[read_features].values

def readability_metric(text, sentence_features, word_usage_features):
    features = []
    tokenized = '\n\n'.join(
        '\n'.join(' '.join(token.value for token in sentence)
                  for sentence in paragraph)
        for paragraph in segmenter.analyze(text))
    results = readability.getmeasures(tokenized, lang='en')
    for sf in sentence_features:
        features.append(results['sentence info'][sf])
    for wf in word_usage_features:
        features.append(results['word usage'][wf])
    return features

def get_readbility_metrics(df):
    sentence_features = ['type_token_ratio', 'syllables', 'long_words', 'complex_words', 'complex_words_dc']
    word_usage_features = ['pronoun', 'tobeverb', 'auxverb']
    features = df[TEXT_COL].apply(lambda x: readability_metric(x, sentence_features, word_usage_features))
    features = np.vstack(features[:])
    return features



 
def train_folds(data):
   
    global params

    if args.get_readability_features:
        # Readability features
        rd_features = get_textstat_readability_features(data)
        data['3k_count'] = data['excerpt'].apply(lambda x: get_3k_feat(x))
        feature_not3k_count = data['3k_count'].values.reshape(data.shape[0],1)

        rd_features_v2 = get_readbility_metrics(df)

        features = np.hstack((rd_features, feature_not3k_count, rd_features_v2))
        print(f'Readability features: {features.shape}')
    # TFIDF features
    if args.get_tfidf:
       tfidf_features = get_tfidf_features(data, max_features=25000)
       features = np.hstack((features, tfidf_features))
    # LASER features
    if args.get_laser:
       laser_features = get_laser_embeddings(data)
       features = np.hstack((features, laser_features))


    #for fold, (train_idx, valid_idx) in enumerate(kf.split(X=data, y=data['bins'].values)):
    for fold in range(5):

       train_data = data[data['kfold']!=fold].reset_index(drop=True)
       valid_data = data[data['kfold']==fold].reset_index(drop=True)

       train_idx = data[data['kfold']!=fold].index.values
       valid_idx = data[data['kfold']==fold].index.values


       #train_data = data.loc[train_idx]
       #valid_data = data.loc[valid_idx]
  
       y_train = train_data[TARGET_COL].values
       y_val = valid_data[TARGET_COL].values

        
       if args.get_bert:
            args.use_single_fc = False
            args.fc_size = 1024
            args.use_pooler = False
            args.use_last_mean = False
            args.use_hidden = False 
            args.use_custom_head = False
            if fold == 0:
               bert_model = '/home/trushant/kaggle/kaggle_clrp/output/electra_l_0620/electra_large_0620/bert_model_fold0.bin'
            elif fold == 1:
               bert_model = '/home/trushant/kaggle/kaggle_clrp/output/electra_l_0620/electra_large_0620/bert_model_fold1.bin'
            elif fold == 2:
               bert_model = '/home/trushant/kaggle/kaggle_clrp/output/electra_l_0620/electra_large_0620/bert_model_fold2.bin'
            elif fold == 3:
               bert_model = '/home/trushant/kaggle/kaggle_clrp/output/electra_l_0620/electra_large_0620/bert_model_fold3.bin'
            elif fold == 4:
               bert_model = '/home/trushant/kaggle/kaggle_clrp/output/electra_l_0620/electra_large_0620/bert_model_fold4.bin'

            bert_features_train = get_bert_predictions(train_data, model_name='google/electra-large-discriminator', model_path=bert_model)
            bert_features_val = get_bert_predictions(valid_data, model_name='google/electra-large-discriminator', model_path=bert_model)
            
            bert_features_train = bert_features_train.reshape(-1, 1) 
            bert_features_val = bert_features_val.reshape(-1, 1)

            if args.use_pca:
                pca = PCA(args.use_pca)
                bert_features_train = pca.fit_transform(bert_features_train)
                bert_features_val = pca.transform(bert_features_val)

            if args.get_readability_features:
                X_train = features[train_idx]
                X_train = np.hstack((X_train, bert_features_train))
                X_val = features[valid_idx]
                X_val = np.hstack((X_val, bert_features_val))
            else:
                X_train = bert_features_train
                X_val = bert_features_val
       else:
            X_train = features[train_idx]
            X_val = features[valid_idx]

       print(f'Feature shape: {X_train.shape}')

       print(f'Fold: {fold}\nTrain: {train_data.shape}\nVal:{valid_data.shape}')
       if args.algo == 'lgb':
          if args.use_autolgb:
               # Use auto lgb on first fold to get hyper-params
              if fold == 0:
                  clf = AutoLGB(objective='regression', metric='rmse', random_state=1234)
                  clf.tune(pd.DataFrame(X_train), pd.DataFrame(y_train))
                  params = clf.params
                  print(f'{params}')
                  print(f'Features: {clf.features}')
          gbm = lgb.LGBMRegressor(**params)
          gbm.fit(X_train, y_train, eval_set=(X_val, y_val), feature_name='auto', categorical_feature = 'auto', verbose=1000)
          val_pred = gbm.predict(X_val)
       elif args.algo == 'xgb':
          gbm = xgb.XGBRegressor(**params)
          gbm.fit(X_train,y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', verbose=1000)
          val_pred = gbm.predict(X_val)
       elif args.algo == 'svm':
          # Scale feature
          scaler = StandardScaler().fit(X_train)
          X_train_std = scaler.transform(X_train)
          model = SVR(C=20,kernel='rbf',gamma='auto')
          model.fit(X_train_std,y_train)
          X_val_std =scaler.transform(X_val)
          val_pred = model.predict(X_val_std)
       elif args.algo == 'ridge':
          model = Ridge() #alpha=1.0) 
          model.fit(X_train,y_train)
          val_pred = model.predict(X_val)

       val_score = np.sqrt(mean_squared_error(y_val, val_pred))
  
       #Dump files to model dir
       with open(op_folder+'params.txt', 'a+') as fid:
         fid.write('\nFold = {} Val RMSE= {}'.format(fold,val_score))

       print('\nFold = {} Val RMSE= {}'.format(fold,val_score))
       print('Saving model..')
       if args.algo == 'lgb':
          gbm.booster_.save_model(op_folder+'gbm_model_fold'+str(fold)+f'_{val_score}.lgb')
       elif args.algo == 'xgb':
          gbm.save_model(op_folder+'gbm_model_fold'+str(fold)+f'_{val_score}.xgb')
       gc.collect()

     

#Dump files to model dir
with open(op_folder+'params.txt', 'w') as fid:
    fid.write(json.dumps(params))

"""    
with open(op_folder+'scores.txt', 'w') as fid:
    fid.write(json.dumps(metrics_dict))

with open(op_folder+'val_aucs.txt', 'w') as fid:
    fid.write(json.dumps(val_auc_dict))
"""

args.use_rd_features = False
df = pd.read_csv('../input/train_folds.csv')
if args.debug:
   df = df.sample(n=100).reset_index(drop=True)

train_folds(df)

