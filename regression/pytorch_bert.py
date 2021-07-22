#/ %% [code]
import platform
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.simplefilter('ignore')
import argparse
import os
import random
import torch.optim as optim
from collections import OrderedDict, defaultdict

import textstat
import pickle
import readability
import syntok.segmenter as segmenter
import wandb
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from torch.autograd.function import InplaceFunction
import math
from torch.utils.data import Sampler,SequentialSampler,RandomSampler,SubsetRandomSampler
from collections import Counter
from sklearn.utils import resample
from torch.optim.swa_utils import AveragedModel, SWALR

parser = argparse.ArgumentParser(description='Process pytorch params.')
parser.add_argument('-model', '--model', type=str, help='Pytorch (timm) model name')
parser.add_argument('-model_dir', '--model_dir', type=str, help='Model save dir name')
parser.add_argument('-folds', type=int, default=5, help='Number of folds')
parser.add_argument('-epochs', '--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('-batch_size', type=int, default=32, help='batch size')
parser.add_argument('-valid_batch_size', type=int, default=32, help='batch size')
parser.add_argument('-test_batch_size', type=int, default=32, help='batch size')
parser.add_argument('-lr', type=float, default=4e-5, help='Learning rate')
parser.add_argument('-lr2', type=float, default=1e-3, help='Learning rate')
parser.add_argument('-loss_fn', type=str, default='mse', help='Loss function')
parser.add_argument('-scheduler', type=str, default='linear', help='Scheduler')
parser.add_argument('-optimizer', type=str, help='Optimizer')
parser.add_argument('-wd', type=float, default=0.01, help='AdamW weight decay')
parser.add_argument('-debug', type=int, help='Number of samples of train data')
parser.add_argument('-max_len', type=int, default='512', help='Max sequence length')
parser.add_argument('-gpu', type=int, default=[0,1], nargs='+', help='CUDA GPUs to use')
parser.add_argument('-use_dropout', type=float, help='Use dropout layer')
parser.add_argument('-multisample_dropout', action='store_true', help='Multi sampled dropout')
parser.add_argument('-lower', action='store_true', help='Lowercase text')
parser.add_argument('-pretrained_model', type=str, help='Model save dir name')
parser.add_argument('-use_hidden', type=str, help='Use BERT hidden layers vs CLS token')
parser.add_argument('-use_pooler', action='store_true', help='Use BERT hidden layers vs CLS token')
parser.add_argument('-use_last_mean', action='store_true', help='Use BERT hidden layers vs CLS token')
parser.add_argument('-use_diff_lr', action='store_true', help='Use differential learning rate')
parser.add_argument('-use_dp', action='store_true', help='Use DataParallel')
parser.add_argument('-hidden_size', type=int, default=768, help='Size of hidden layer')
parser.add_argument('-accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('-eval_steps', type=int, help='Gradient accumulation steps')
parser.add_argument('-seed', type=int, default=1234, help='seed for repro')
parser.add_argument('-train_file', type=str, default='../input/train_folds.csv', help='seed for repro')
parser.add_argument('-sample_target', action='store_true', help='Sample target')
parser.add_argument('-use_rd_features', action='store_true', help='Sample target')
parser.add_argument('-use_single_fc', action='store_true', help='Sample target')
parser.add_argument('-fc_size', type=int, default=1024*3, help='Sample target')
parser.add_argument('-run_name', type=str, default='test-run', help='Wandb run name')
parser.add_argument('-enable_wandb', action='store_true', help='Enable logging in wandb')
parser.add_argument('-run_folds', type=int, nargs='+', help='Run targetted folds')
parser.add_argument('-use_custom_head', action='store_true', help='Use Custom Head')
parser.add_argument('-init_weights', action='store_true', help='Init weights')
parser.add_argument('-reinit_weights', type=int, help='Re-init last n layers of transformer')
parser.add_argument('-smooth_loss', action='store_true', help='Smooth loss')
parser.add_argument('-mixout', type=float, default=0, help='Mixout')
parser.add_argument('-use_sampler', action='store_true', help='Smooth loss')
parser.add_argument('-dynamic_padding', action='store_true', help='Smooth loss')
parser.add_argument('-grad_clip', type=float, help='Gradient Clipping')
parser.add_argument('-upsample', type=float, help='Upsampling minority')
parser.add_argument('-no_split_proc', action='store_true', help='no split in preproc')
parser.add_argument('-automodel_seq', action='store_true', help='Use AutoModelForSequenceClassification')
parser.add_argument('-no_amp', action='store_true', help='Disable AMP')
parser.add_argument('-use_conv_head', action='store_true', help='Disable AMP')


best_loss = 100
args = parser.parse_args()

if args.enable_wandb:
    wandb.init(entity='kaggle-clrp', name=args.run_name, project='CommonlitReadabilityTrain')

TEXT_COL = 'excerpt'
TARGET_COL = 'target'

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)
print(f'Using CUDA Devices: {os.environ["CUDA_VISIBLE_DEVICES"]}')

if args.model_dir:
    model_dir = args.model_dir
    SAVE_MODEL_FOLDER = '../output/' + model_dir
    if not os.path.exists(SAVE_MODEL_FOLDER):
        os.makedirs(SAVE_MODEL_FOLDER)


# %% [markdown]
# We define a Config class to store variables and functions that are to be used globally inside our training script.
# This makes the code more modular and easy to approach at the same time.

# %% [code]
class Config:
    seed = args.seed
    NB_EPOCHS = args.epochs
    LR = args.lr
    MAX_LEN = args.max_len
    TRAIN_BS = args.batch_size
    VALID_BS = args.valid_batch_size
    BERT_MODEL = args.model
    DBERT_MODELS = ['distilbert-base-uncased', 'xlnet', 't5', 'valhalla/bart-large-finetuned-squadv1', 'facebook/bart-large', 't5-large', 'facebook/dpr-reader-multiset-base']
    FILE_NAME = args.train_file #'../input/train_folds.csv'
    TOKENIZER = transformers.AutoTokenizer.from_pretrained(BERT_MODEL,return_tensors='pt')
    scaler = GradScaler()

params_file = os.path.join(SAVE_MODEL_FOLDER, 'params.txt')
with open(params_file, 'w') as f:
        f.write('Arguments:\n')
        attrs = vars(args)
        f.write(', '.join("%s: %s" % item for item in attrs.items()))
        f.write('\n\n')
        f.write('Config:\n')
        attrs = vars(Config)
        f.write(', '.join("%s: %s" % item for item in attrs.items()))
        f.write('\n\n')

# Torch utils
def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def sample_target(features, target):
    mean, stddev = target
    sampled_target = tf.random.normal([], mean=tf.cast(mean, dtype=tf.float32), 
                                      stddev=tf.cast(stddev, dtype=tf.float32), dtype=tf.float32)
    
    return (features, sampled_target)

def get_readability_features(df):
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


def get_3k_feat(text):
    with open('./3k_words.pkl', 'rb') as f:
        words_3k = pickle.load(f)
    count = 0
    for word in text.split(' '):
        if word not in words_3k:
            count += 1
    return count


def get_fold_loader(train_data=None, sampler=None,indices=None):
    train_ds = train_data
    
    if sampler=='random':
        
        train_sampler = RandomSampler(train_ds,replacement=True)
        train_dataloader = DataLoader(train_ds,
                                     sampler = train_sampler,
                                     batch_size=Config.TRAIN_BS,
                                     drop_last=False)
        
        
    
    elif sampler=='sequential':
        
        train_sampler = SequentialSampler(train_ds)
        train_dataloader = DataLoader(train_ds,
                                     sampler = train_sampler,
                                     batch_size=Config.TRAIN_BS,
                                     drop_last=False)
        
    elif sampler=='subset':
        
        train_sampler = SubsetRandomSampler(indices)
        train_dataloader = DataLoader(train_ds,
                                     sampler = train_sampler,
                                     batch_size=Config.TRAIN_BS,
                                     drop_last=False)
        
        
        
        
        
    elif sampler=="weighted":
        
        train_sampler = weightedsampler(train_ds)
        train_dataloader = DataLoader(train_ds,
                                     sampler = train_sampler,
                                     batch_size=Config.TRAIN_BS,
                                     num_workers=8,
                                     worker_init_fn=seed_worker,
                                     drop_last=False)
        
    else:
        train_dataloader = DataLoader(train_ds,
                                      shuffle=True,
                                     batch_size=Config.TRAIN_BS,
                                     drop_last=False)
    
    
    return train_dataloader
        

class weightedsampler(Sampler):
    
    def __init__(self,dataset):
        
        self.indices = list(range(len(dataset)))
        self.num_samples = len(dataset)
        self.label_to_count = dict(Counter(dataset.bins))
        weights = [1/self.label_to_count[i] for i in dataset.bins]
        
        self.weights = torch.tensor(weights,dtype=torch.double)
        
    def __iter__(self):
        count = 0
        index = [self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True)]
        while count < self.num_samples:
            yield index[count]
            count += 1
    
    def __len__(self):
        return self.num_samples
        

class CLMCollate:
    
    def __init__(self):
        self.seq_dic = defaultdict(int)  ## used to track max_length
        self.batch_record = defaultdict(list)
        self.bn = 0
        
    def __call__(self,batch):
        out = {'ids' :[],
               'mask':[],
               'token_type_ids':[],
               'targets':[],
               'errors': [],
               'rd_features': [],
               'bins': []
        }
        
        for i in batch:
            for k,v in i.items():
                out[k].append(v)
                
        if args.dynamic_padding:
            max_pad =0
            
            for p in out['ids']:
                if max_pad < len(p):
                    max_pad = len(p)
                    
        else:
            max_pad = args.max_len
            
        
        self.batch_record[str(self.bn)] = [len(x) for x in out['ids']]  
        self.seq_dic[str(self.bn)] = max_pad
        self.bn+=1
        for i in range(len(batch)):
            input_id = out['ids'][i]
            att_mask = out['mask'][i]
            token_type_id = out['token_type_ids'][i]
            text_len = len(input_id)

            # Add pad based on text len in batch
            out['ids'][i] = np.hstack((out['ids'][i].detach().numpy(), [1] * (max_pad - text_len))[:max_pad])
            out['mask'][i] = np.hstack((out['mask'][i].detach().numpy(), [0] * (max_pad - text_len))[:max_pad])
            out['token_type_ids'][i] = np.hstack((out['token_type_ids'][i].detach().numpy(), [0] * (max_pad - text_len))[:max_pad])
              
        out['ids'] = torch.tensor(out['ids'],dtype=torch.long)
        out['mask'] = torch.tensor(out['mask'],dtype=torch.long)
        out['token_type_ids'] = torch.tensor(out['token_type_ids'],dtype=torch.long)
        out['targets'] = torch.tensor(out['targets'],dtype=torch.float)
        out['errors'] = torch.tensor(out['errors'],dtype=torch.float)
        out['rd_features'] = torch.tensor(out['rd_features'],dtype=torch.float)
 
        return out


# %% [code]
class BERTDataset(Dataset):
    def __init__(self, text, rd_features= None, target=None, errors=None, is_test=False, bins=None):
        self.text = text
        self.target = target
        self.errors = errors
        self.is_test = is_test
        self.tokenizer = Config.TOKENIZER
        self.max_len = Config.MAX_LEN
        self.rd_features = rd_features
        self.bins = bins

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        text = str(self.text[idx])
        rd_features = self.rd_features[idx]
        if args.lower:
           text = text.lower()
        #text = text.replace('\n', '')
        if not args.no_split_proc:
            text = ' '.join(text.split())
        global inputs
        if not self.is_test:
            bins = self.bins[idx]

        if args.dynamic_padding:
            inputs = self.tokenizer.encode_plus(
                text=text,
                truncation=False,
                add_special_tokens=True,
                padding=False,
                return_attention_mask=True,
                return_token_type_ids=True
            ) 
        else:
            if 'gpt2' in args.model:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer.encode_plus(
                text=text,
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
            standard_errors = torch.tensor(self.errors[idx])

            if args.sample_target:
            	sample_target()

            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
                'targets': targets,
                'errors': standard_errors,
                'rd_features': rd_features,
                'bins': bins
            }


# %% [markdown]
# Below is a custom `Trainer` class that I wrote from scratch to facilitate my training and validation sub-routines.
# 
# This class hence provides a very "fastai" type interface for doing training.


def KLDivLoss(outputs, targets, standard_error):
   output_mean = torch.mean(outputs)
   output_std = torch.std(outputs)

   target_mean = torch.mean(targets)

   p = torch.distributions.Normal(output_mean, output_std)
   q = torch.distributions.Normal(target_mean, standard_error)
   loss = torch.distributions.kl_divergence(p, q)
   loss = loss.mean()
   return loss


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


def loss_with_error(output,target,error):
    l_params = 0.1
    leaky_func = nn.LeakyReLU(l_params)
    return torch.mean(leaky_func((output-target)**2-error**2/4)+l_params*(error**2/4))/2


# %% [code]
class Trainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        scheduler, 
        train_dataloader, 
        valid_dataloader,
        device
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.loss_fn = self.yield_loss
        self.device = device
        self.loss_str = ''

    def yield_loss(self, outputs, targets, errors):
        """
        This is the loss function for this task
        """
        if args.loss_fn == 'mse':
            return torch.sqrt(nn.MSELoss()(outputs, targets))
        elif args.loss_fn == 'kldiv':
            return KLDivLoss(outputs, targets, errors)
        elif args.loss_fn == 'huber':
            return torch.nn.SmoothL1Loss()(outputs, targets)
        elif args.loss_fn == 'logcosh': 
            return LogCoshLoss()(outputs, targets)
        elif args.loss_fn == 'custom':
            return loss_with_error(outputs, targets, errors)

    def train_one_epoch(self):
        global best_loss
        """
        This function trains the model for 1 epoch through all batches
        """
        prog_bar = tqdm(enumerate(self.train_data), total=len(self.train_data))
        self.model.train()
        if not args.no_amp:
          with autocast():
            for step, inputs in prog_bar:
                ids = inputs['ids'].to(self.device, dtype=torch.long)
                mask = inputs['mask'].to(self.device, dtype=torch.long)
                ttis = inputs['token_type_ids'].to(self.device, dtype=torch.long)
                targets = inputs['targets'].to(self.device, dtype=torch.float)
                errors = inputs['errors'].to(self.device, dtype=torch.float)
                rd_features = inputs['rd_features'].to(self.device, dtype=torch.float)
                # Forward pass of model
                if any(x in Config.BERT_MODEL for x in Config.DBERT_MODELS):
                   ttis = None
                outputs = self.model(ids=ids, mask=mask, token_type_ids=ttis, rd_features=rd_features)
                loss = self.loss_fn(outputs, targets, errors)
                loss = loss / args.accumulation_steps

                prog_bar.set_description('Train loss: {:.2f}'.format(loss.item()))

                if args.enable_wandb:
                    wandb.log({"train_loss": loss})

                Config.scaler.scale(loss).backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)

                if ((step + 1) % args.accumulation_steps == 0) or ((step + 1) == len(self.train_data)):
                    Config.scaler.step(self.optimizer)
                    Config.scaler.update()
                    self.optimizer.zero_grad()

                    # Moved scheduler step irrespective of grad accum. Want this to change per step
                    if args.scheduler== 'plateau':
                        self.scheduler.step(loss)
                    else:
                        if args.scheduler == 'swa':
                            self.model.update_parameters(self.model)
                        self.scheduler.step()

                # Eval every eval_steps if specified
                if args.eval_steps and (((step + 1) % args.eval_steps == 0) or ((step + 1) == len(self.train_data))):
                    current_loss = self.valid_one_epoch()
                    if args.enable_wandb:
                        wandb.log({"val_loss": loss})
                    if current_loss < best_loss:
                        #print(f'Current loss: {current_loss} Best loss: {best_loss}')
                        print(f"Saving best model in this fold: {current_loss:.4f}")
                        self.loss_str = str(current_loss).replace('.','_')
                        model = self.get_model()
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = f"{SAVE_MODEL_FOLDER}/bert_model_fold{fold}.bin" #_epoch{epoch}_{self.loss_str}.bin"
                        torch.save(model_to_save.state_dict(), output_model_file)
                        best_loss = current_loss

        else:
            for step, inputs in prog_bar:
                ids = inputs['ids'].to(self.device, dtype=torch.long)
                mask = inputs['mask'].to(self.device, dtype=torch.long)
                ttis = inputs['token_type_ids'].to(self.device, dtype=torch.long)
                targets = inputs['targets'].to(self.device, dtype=torch.float)
                errors = inputs['errors'].to(self.device, dtype=torch.float)
                rd_features = inputs['rd_features'].to(self.device, dtype=torch.float)
                # Forward pass of model
                if any(x in Config.BERT_MODEL for x in Config.DBERT_MODELS):
                   ttis = None
                outputs = self.model(ids=ids, mask=mask, token_type_ids=ttis, rd_features=rd_features)         
                loss = self.loss_fn(outputs, targets, errors)
                loss = loss / args.accumulation_steps
                
                prog_bar.set_description('Train loss: {:.2f}'.format(loss.item()))
           
                if args.enable_wandb:
                    wandb.log({"train_loss": loss})

                #Config.scaler.scale(loss).backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)

                if ((step + 1) % args.accumulation_steps == 0) or ((step + 1) == len(self.train_data)):
                    #self.optimizer.step()
                    #self.optimizer.zero_grad()
              
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # Moved scheduler step irrespective of grad accum. Want this to change per step
                    if args.scheduler== 'plateau':
                        self.scheduler.step(loss)
                    else:
                        if args.scheduler == 'swa':
                            self.model.update_parameters(self.model)
                        self.scheduler.step()

                # Eval every eval_steps if specified
                if args.eval_steps and (((step + 1) % args.eval_steps == 0) or ((step + 1) == len(self.train_data))):
                    current_loss = self.valid_one_epoch()
                    if args.enable_wandb:
                        wandb.log({"val_loss": loss})
                    if current_loss < best_loss:
                        #print(f'Current loss: {current_loss} Best loss: {best_loss}')
                        print(f"Saving best model in this fold: {current_loss:.4f}")
                        self.loss_str = str(current_loss).replace('.','_')
                        model = self.get_model()
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = f"{SAVE_MODEL_FOLDER}/bert_model_fold{fold}.bin" #_epoch{epoch}_{self.loss_str}.bin"
                        torch.save(model_to_save.state_dict(), output_model_file)
                        best_loss = current_loss
                   
        params_file = os.path.join(SAVE_MODEL_FOLDER, 'params.txt')
        with open(params_file, 'a+') as f:
           f.write('Train loss: {:.2f} '.format(loss.item()))
 
    def valid_one_epoch(self):
        """
        This function validates the model for one epoch through all batches of the valid dataset
        It also returns the validation Root mean squared error for assesing model performance.
        """
        prog_bar = tqdm(enumerate(self.valid_data), total=len(self.valid_data))
        self.model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for idx, inputs in prog_bar:
                ids = inputs['ids'].to(self.device, dtype=torch.long)
                mask = inputs['mask'].to(self.device, dtype=torch.long)
                ttis = inputs['token_type_ids'].to(self.device, dtype=torch.long)
                targets = inputs['targets'].to(self.device, dtype=torch.float)
                errors = inputs['errors'].to(self.device, dtype=torch.float)
                rd_features = inputs['rd_features'].to(self.device, dtype=torch.float)
                if any(x in Config.BERT_MODEL for x in Config.DBERT_MODELS):
                   ttis = None
                outputs = self.model(ids=ids, mask=mask, token_type_ids=ttis, rd_features=rd_features)

                if args.smooth_loss:
                    mu, sigma = norm.fit(np.concatenate([outputs.cpu().numpy()], 0) - np.concatenate([targets.cpu().numpy()], 0))
                    loss = self.loss_fn(outputs-mu, targets, errors)
                else:
                    loss = self.loss_fn(outputs, targets, errors)
                if args.enable_wandb:
                    wandb.log({"val_loss": loss})
                prog_bar.set_description('Val loss: {:.2f}'.format(loss.item()))
                all_targets.extend(targets.cpu().detach().numpy().tolist())
                all_predictions.extend(outputs.cpu().detach().numpy().tolist())
        val_rmse_loss = np.sqrt(mean_squared_error(all_targets, all_predictions))
        params_file = os.path.join(SAVE_MODEL_FOLDER, 'params.txt')
        with open(params_file, 'a+') as f:
            f.write('Val loss: {:.2f} RMSE: {:.3f}\n'.format(loss.item(),val_rmse_loss))
        print('Validation RMSE: {:.3f}'.format(val_rmse_loss))
        
        if args.enable_wandb:
            wandb.log({"val_rmse": loss})

        return val_rmse_loss
    
    def get_model(self):
        return self.model



class Mixout(InplaceFunction):
    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input, target=None, p=0.0, training=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("A mix probability of mixout has to be between 0 and 1," " but got {}".format(p))
        if target is not None and input.size() != target.size():
            raise ValueError(
                "A target tensor size must match with a input tensor size {},"
                " but got {}".format(input.size(), target.size())
            )
        ctx.p = p
        ctx.training = training

        if ctx.p == 0 or not ctx.training:
            return input

        if target is None:
            target = cls._make_noise(input)
            target.fill_(0)
        target = target.to(input.device)

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        ctx.noise = cls._make_noise(input)
        if len(ctx.noise.size()) == 1:
            ctx.noise.bernoulli_(1 - ctx.p)
        else:
            ctx.noise[0].bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise[0].repeat(input.size()[0], 1)
        ctx.noise.expand_as(input)

        if ctx.p == 1:
            output = target
        else:
            output = ((1 - ctx.noise) * target + ctx.noise * output - ctx.p * target) / (1 - ctx.p)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.training:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None


def mixout(input, target=None, p=0.0, training=False, inplace=False):
    return Mixout.apply(input, target, p, training, inplace)


class MixLinear(torch.nn.Module):
    __constants__ = ["bias", "in_features", "out_features"]
    def __init__(self, in_features, out_features, bias=True, target=None, p=0.0):
        super(MixLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.target = target
        self.p = p

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, mixout(self.weight, self.target, self.p, self.training), self.bias)

    def extra_repr(self):
        type = "drop" if self.target is None else "mix"
        return "{}={}, in_features={}, out_features={}, bias={}".format(
            type + "out", self.p, self.in_features, self.out_features, self.bias is not None
        )

# Model

class BERTModel(nn.Module):
    def __init__(self, multisample_dropout=True):
        super(BERTModel, self).__init__()
        if args.use_rd_features:
            self.rd_feature_len = 21
            self.layer_norm = nn.LayerNorm(args.fc_size + self.rd_feature_len)
        else:
            self.rd_feature_len = 0

        if args.use_last_mean:
            pass
            #self.layer_norm = nn.LayerNorm(args.fc_size)
        self.config = transformers.AutoConfig.from_pretrained(Config.BERT_MODEL) 
        if 't5' in Config.BERT_MODEL:
            self.bert = transformers.T5EncoderModel.from_pretrained(Config.BERT_MODEL)
            self.pooler = T5Pooler(args.hidden_size)
        else:
            if args.pretrained_model:
                print(f'Loading pretrained model: {args.pretrained_model}') 
                self.bert = transformers.AutoModel.from_pretrained(args.pretrained_model, output_hidden_states=True)
                if args.automodel_seq: 
                    self.bert = transformers.AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels = 1, output_hidden_states=False, output_attentions=False)
                
            else:
                if 'dpr' in Config.BERT_MODEL:
                    self.bert = transformers.DPRReader.from_pretrained(Config.BERT_MODEL , output_hidden_states=True)
                elif args.automodel_seq:
                    self.bert = transformers.AutoModelForSequenceClassification.from_pretrained(Config.BERT_MODEL, num_labels = 1, output_hidden_states=False, output_attentions=False)
                else:
                    self.bert = transformers.AutoModel.from_pretrained(Config.BERT_MODEL , output_hidden_states=True)

        if args.use_dropout and (args.use_single_fc):
           if multisample_dropout:
               self.dropouts = nn.ModuleList([
                 nn.Dropout(args.use_dropout) for _ in range(5)
               ])
           else:
              self.dropouts = nn.ModuleList([nn.Dropout(args.use_dropout)])
    
        # Custom head
        if args.use_single_fc:
            self.fc = nn.Linear(args.fc_size + self.rd_feature_len, 1)
        elif args.use_custom_head:
            self.attention = nn.Sequential(            
                nn.Linear(args.fc_size, 512),            
                nn.Tanh(),                       
                nn.Linear(512, 1),
                nn.Softmax(dim=1)
            )        
            
            self.regressor = nn.Sequential(                        
                OrderedDict([
                    ('dropout0', nn.Dropout(args.use_dropout)),
                    ('fc', nn.Linear(args.fc_size, 1))                        
                 ])
                )
        elif args.automodel_seq:
            pass
        elif args.use_conv_head:
            self.conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv1d(args.fc_size, 256, kernel_size=3)),
            ('act1', nn.GELU()),
            ('dropout1', nn.Dropout(args.use_dropout)),
            ]))
            #self.conv = nn.Conv1d(args.fc_size, 256, kernel_size=2)
            self.fc = nn.Linear(256, 1)
        else:
            self.whole_head = nn.Sequential(OrderedDict([
            ('dropout0', nn.Dropout(args.use_dropout)),
            ('l1', nn.Linear(args.fc_size + self.rd_feature_len, 256)),
            ('act1', nn.GELU()),
            ('dropout1', nn.Dropout(args.use_dropout)),
            ('l2', nn.Linear(256, 1))
            ]))

        if args.init_weights:
            self._init_weights(self.fc)

        if args.mixout > 0:
            print(f'Initializing Mixout Regularization: Mixup={args.mixout}')
            for sup_module in self.bert.modules():
                for name, module in sup_module.named_children():
                    if isinstance(module, nn.Dropout):
                        module.p = 0.0
                    if isinstance(module, nn.Linear):
                        target_state_dict = module.state_dict()
                        bias = True if module.bias is not None else False
                        new_module = MixLinear(
                           module.in_features, module.out_features, bias, target_state_dict["weight"], args.mixout
                        )
                        new_module.load_state_dict(target_state_dict)
                        setattr(sup_module, name, new_module)
            gc.collect()
            torch.cuda.empty_cache()
            print('Done!')

        if args.reinit_weights:
            reinit_layers = args.reinit_weights
            if 'bert' in args.model:
                print(f'Reinitializing Last {reinit_layers} Layers of the Transformer Model')
                #encoder_temp = getattr(self.bert, _model_type)
                for layer in self.bert.encoder.layer[-reinit_layers:]:
                    for module in layer.modules():
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
                print('Done!')
            elif 'bart' in args.model or 'funnel' in args.model:
               print(f'Reinitializing Last {reinit_layers} Layers ...')
               for layer in self.bert.decoder.layers[-reinit_layers :]:
                   for module in layer.modules():
                        self.bert._init_weights(module)
               print('Done.!')

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
    
    def forward(self, ids, mask, rd_features, token_type_ids=None):
        
        # Returns keys(['last_hidden_state', 'pooler_output', 'hidden_states'])
        if token_type_ids is not None:
            output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=True)
           # output = self.bert(ids, attention_mask=mask, return_dict=True)
        else:
            output = self.bert(ids, attention_mask=mask, return_dict=True)
    
        #print(output.keys())

        #output = self.bert(ids, return_dict=True)
     

        # Hidden layer
        if args.use_hidden: 
          if args.use_hidden == 'last':
              # Last  hidden states
              if 'bart' in Config.BERT_MODEL:
                  output = output['decoder_hidden_states'][-1]
              else:
                  output = output['hidden_states'][-1]
              if args.use_custom_head:
                  weights = self.attention(output)
                  output = torch.sum(weights * output, dim=1)        
              else:
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
          elif args.use_hidden == 'conv':
              hs = output['hidden_states']
              seq_output = torch.cat([hs[-1],hs[-2],hs[-3]], dim=-1)
              input_mask_expanded = mask.unsqueeze(-1).expand(seq_output.size()).float()
              sum_embeddings = torch.sum(seq_output * input_mask_expanded, 1)
              sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
              output = sum_embeddings / sum_mask
              output = output.reshape(-1, 3, 1024)
              output = output.permute(0,2,1)
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
          #output = self.layer_norm(output)

          if args.use_rd_features:
              output = torch.cat((output, rd_features),1)
              output = self.layer_norm(output)
        elif args.automodel_seq:
            if 't5' in args.model:
                output = output['last_hidden_state']
            else:
                output = output['logits']
        # CLS
        else:
          # Last layer
          if 'dpr' in Config.BERT_MODEL:
              output = output['end_logits']
          else:
              output = output['last_hidden_state']
              # CLS token
              output = output[:,0,:]
          if args.use_rd_features:
              output = torch.cat((output, rd_features),1)
              output = self.layer_norm(output)
         
          # Convert to 3 channel for Conv layer
          #output = output.unsqueeze(1).repeat(1,3,1)
          #output = output.permute(0,2,1)

        #output = self.layer_norm(output)

        #print(f'Output: {output}')
        #if args.debug:
         #   print(output.shape)

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
                output = self.regressor(output) #self.custom_head(output)
            elif args.automodel_seq:
                pass
            elif args.use_conv_head:
                output = self.conv(output)
                output = output.squeeze()
                output = self.fc(output)
            else:
                output = self.whole_head(output) 

        output = output.squeeze(-1).squeeze(-1)
        return output


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector


class T5Pooler(nn.Module):
    def __init__(self, hidden_size, activation=nn.Tanh()):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = activation
        
    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def create_optimizer(model):
    if args.debug:
        for i, layer in enumerate(model.named_parameters()):
            print(i, layer[0])
    model_init_lr = args.lr
    multiplier = 0.975
    classifier_lr = args.lr
    parameters = []

    # Encoders
    lr = model_init_lr
    end_layer = 23
    for layer in range(end_layer,-1,-1):
        """ 
        # For Funnel arch
        layer0 = str(layer)[0]
        try:
            layer1 = str(layer)[1]
        except Exception as e:
            layer1 = layer0
            layer0 = 0
        #print(f'{layer0}.{layer1}')
        # End funnel arch
        """
        layer_params = {
            'params': [p for n,p in model.named_parameters() if f'encoder.layer.{layer}.' in n],
        #     'params': [p for n,p in model.named_parameters() if f'encoder.blocks.{layer0}.{layer1}' in n],  # funnel arch
            'lr': lr
        }
        parameters.append(layer_params)
        lr *= multiplier

    # Decoders
    """
    lr = model_init_lr
    end_layer = 2 #23
    for layer in range(end_layer,-1,-1):
        # End funnel arch
        layer_params = {
             'params': [p for n,p in model.named_parameters() if f'decoder.layers.{layer}' in n],  # funnel arch
            'lr': lr
        }
        parameters.append(layer_params)
        lr *= multiplier
 
    """

    # Head
    classifier_params = {
        'params': [p for n,p in model.named_parameters() if 'conv' in n or 'fc' in n or 'pooler' in n or 'whole_head' in n],
        'lr': classifier_lr
    }
    parameters.append(classifier_params)

    return  transformers.AdamW(parameters)


def create_optimizer_old(model):
    named_parameters = list(model.named_parameters())    
    if args.debug:
        for i, layer in enumerate(named_parameters):
            print(i, layer[0])
     
    #print(len(named_parameters)) 
    #print(named_parameters)
    bert_parameters = named_parameters[:390]    
    regressor_parameters = named_parameters[391:]
        
    regressor_group = [params for (name, params) in regressor_parameters]

    parameters = []
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Standard lr/wd for non bert layers
    parameters.append({
        "params": [  p for n, p in regressor_parameters if not any(nd in n for nd in no_decay)],
        "weight_decay": args.wd,
        "lr": args.lr
        })

    parameters.append({
        "params": [ p for n, p in regressor_parameters if any(nd in n for nd in no_decay)],
        "weight_decay": 0,
        "lr": args.lr
        })


    # diff lr for bert layers
    for layer_num, (name, params) in enumerate(bert_parameters):
        weight_decay = 0.0 if "bias" in name else args.wd

        lr = args.lr/2.6

        # layer 20 till decoder
        if layer_num >= 165: #69        
            lr = args.lr

        # trying start of decoder till end 
        if layer_num >= 341: #133 
            lr = 2.6*args.lr

        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})

    return transformers.AdamW(parameters)


def get_optimizer(model):
    # differential learning rate and weight decay
    param_optimizer = list(model.named_parameters())
    learning_rate = args.lr
    no_decay = ['bias', 'gamma', 'beta']

    group1=['layer.0.','layer.1.','layer.2.','layer.3.']
    group2=['layer.4.','layer.5.','layer.6.','layer.7.']    
    group3=['layer.8.','layer.9.','layer.10.','layer.11.']
    group_all=['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.','layer.8.','layer.9.','layer.10.','layer.11.']
    optimizer_parameters = [
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': args.wd},
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': args.wd, 'lr': learning_rate/2.6},
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': args.wd, 'lr': learning_rate},
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': args.wd, 'lr': learning_rate*2.6},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': 0.0},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.0, 'lr': learning_rate/2.6},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.0, 'lr': learning_rate},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.0, 'lr': learning_rate*2.6},
        {'params': [p for n, p in model.named_parameters() if "bert" not in n], 'lr':args.lr2, "momentum" : 0.99},
    ]
    return transformers.AdamW(optimizer_parameters, lr=args.lr)


# %% [code]
def yield_optimizer(model):
    """
    Returns optimizer for specific parameters
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.wd, #0.003
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return transformers.AdamW(optimizer_parameters, lr=Config.LR)

def get_bert_predictions(test_data, model_path):
        print('Getting BERT Embeddings')
        """
        This function validates the model for one epoch through all batches of the valid dataset
        It also returns the validation Root mean squared error for assesing model performance.
        """
        model = BERTModel(multisample_dropout=args.multisample_dropout).to(DEVICE)
        model.to(DEVICE)
        model.load_state_dict(torch.load(model_path), strict=True)


        num_bins = 5
        test_data.loc[:,'bins'] = pd.cut(test_data['target'],bins=num_bins,labels=False)
        test_set = BERTDataset(
            text = test_data[TEXT_COL].values,
            target = test_data[TARGET_COL].values,
            errors = test_data['standard_error'].values,
            rd_features = get_rd_features(test_data),
            bins =  test_data['bins'].values
        )

        if args.dynamic_padding:
            sequence = CLMCollate()
            test_data_loader = DataLoader(
                test_set,
                batch_size = args.test_batch_size,
                collate_fn=sequence,
                shuffle = False,
                num_workers=8,
                worker_init_fn=seed_worker,
                drop_last=False
            )
        else: 
            test_data_loader = DataLoader(
                test_set,
                batch_size = args.test_batch_size,
                shuffle = False,
                num_workers=8,
                drop_last=False
            )

        prog_bar = tqdm(enumerate(test_data_loader), total=len(test_data_loader))
        model.eval()
        all_predictions = []
        with torch.no_grad():
            for idx, inputs in prog_bar:
                ids = inputs['ids'].to(DEVICE, dtype=torch.long)
                mask = inputs['mask'].to(DEVICE, dtype=torch.long)
                ttis = inputs['token_type_ids'].to(DEVICE, dtype=torch.long)
                rd_features = inputs['rd_features'].to(DEVICE, dtype=torch.float)
                if any(x in Config.BERT_MODEL for x in Config.DBERT_MODELS):
                   ttis = None
                outputs = model(ids=ids, mask=mask, rd_features=rd_features, token_type_ids=ttis)
                all_predictions.extend(outputs.cpu().detach().numpy())

        return all_predictions


def get_rd_features(df):
    rd_features = get_readability_features(df)
    df['3k_count'] = df['excerpt'].apply(lambda x: get_3k_feat(x))
    feature_not3k_count = df['3k_count'].values.reshape(df.shape[0],1)
    rd_features_v2 = get_readbility_metrics(df)
    rd_features = np.hstack((rd_features, feature_not3k_count, rd_features_v2))
    return rd_features

# Main training loop
if __name__ == '__main__':
    seed_everything(Config.seed)

    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
        DEVICE = torch.device('cuda')
    else:
        print("\n[INFO] GPU not found. Using CPU: {}\n".format(platform.processor()))
        DEVICE = torch.device('cpu')
    
    data = pd.read_csv(Config.FILE_NAME)
   
    if args.debug:
       data = data.sample(n=args.debug)

    oof_predictions = np.zeros(len(data))
 
    for fold in range(5):
        # Optionally skip certain folds based on user args 
        if args.run_folds and fold not in args.run_folds:
            print(f'Skipping fold {fold}')
            continue

        print(f"Fold: {fold}")
        print('-'*20)
        best_loss = 100
        #print(f'Best loss: {best_loss}')

        # Create fold data
        valid_idx = data[data['kfold']==fold].index.values
        train_data = data[data['kfold']!=fold].reset_index(drop=True)
        valid_data = data[data['kfold']==fold].reset_index(drop=True)

        num_bins = 5
        train_data.loc[:,'bins'] = pd.cut(train_data['target'],bins=num_bins,labels=False)
        valid_data.loc[:,'bins'] = pd.cut(valid_data['target'],bins=num_bins,labels=False)


    # Upsample train set to prevent leakage in val set
        if args.upsample:
            
            # Initialize with majority class
            final_df = train_data[train_data[TARGET_COL] <= 0 ]
            print(f'Maj class: {final_df.shape}') 
             # number of samples in majority class
            samples = int(args.upsample * final_df.shape[0])
            print(f"Sampling to match {samples} samples")
            df_train_min = train_data[train_data[TARGET_COL] > 0]
            print(f'min class shape: {df_train_min.shape}')
            upsample_df = resample(df_train_min,
                                   replace=True,  # sample without replacement
                                   n_samples=samples,  # to match minority class
                                   random_state=123)
            final_df = pd.concat([final_df, upsample_df])
            train_data = final_df
            print(train_data.shape)
        train_set = BERTDataset(
            text = train_data['excerpt'].values,
            target = train_data['target'].values,
            errors = train_data['standard_error'].values,
            rd_features = get_rd_features(train_data),
            bins = train_data['bins'].values
        )

        valid_set = BERTDataset(
            text = valid_data['excerpt'].values,
            target = valid_data['target'].values,
            errors = valid_data['standard_error'].values,
            rd_features = get_rd_features(valid_data),
            bins = valid_data['bins'].values
        )

        if not args.use_sampler:
            if args.dynamic_padding:
                sequence = CLMCollate()
                train_loader = DataLoader(
                    train_set,
                    batch_size = Config.TRAIN_BS,
                    collate_fn=sequence,
                    shuffle = True,
                    num_workers=8,
                    worker_init_fn=seed_worker,
                    drop_last=True
                )
            else:
                train_loader = DataLoader(
                    train_set,
                    batch_size = Config.TRAIN_BS,
                    shuffle = True,
                    num_workers=8,
                    worker_init_fn=seed_worker,
                    drop_last=True
                )
        else:
            train_loader = get_fold_loader(train_data=train_set, sampler='weighted')

        if args.dynamic_padding:
           sequence = CLMCollate()
           valid_loader = DataLoader(
                valid_set,
                batch_size = Config.VALID_BS,
                collate_fn=sequence,
                shuffle = False,
                num_workers=8,
                worker_init_fn=seed_worker,
                drop_last=True
           )
        else:
           valid_loader = DataLoader(
                valid_set,
                batch_size = Config.VALID_BS,
                shuffle = False,
                num_workers=8,
                worker_init_fn=seed_worker,
                drop_last=True
           )

        model = BERTModel(multisample_dropout=args.multisample_dropout).to(DEVICE)
        #model = model.half()
        print(model)
        print(f'Scheduler: {args.scheduler} Learning Rate: {args.lr}')
        print(f'Loss function: {args.loss_fn}')
        if args.use_diff_lr:
            print(f'Using differential learning rate per layer')
 
        # Architectures
        if args.use_hidden:
            print(f'Using hidden layers for model output. Strategy: {args.use_hidden}')
        elif args.use_pooler:
            print('Using pooler for model output')
        elif args.use_last_mean:
            print('Using mean of last layer')
        else:
            print('Using CLS token for model output')

        nb_train_steps = int(len(train_data) / Config.TRAIN_BS * Config.NB_EPOCHS)
       
        # Differential LR 
        if args.use_diff_lr:
            optimizer = create_optimizer(model)
        else:
            optimizer = yield_optimizer(model)

        if args.scheduler == 'linear':
           scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=nb_train_steps
           )
        elif args.scheduler == 'cosine':
            scheduler = transformers.get_cosine_schedule_with_warmup( #get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=nb_train_steps,
            num_cycles=1
           )
        elif args.scheduler == 'step':
            scheduler = optim.lr_scheduler.MultiStepLR(
              optimizer,
              milestones=[30, 60, 90],
              gamma=0.1
            )
        elif args.scheduler == 'constant':
           scheduler = transformers.get_constant_schedule(optimizer)
        elif args.scheduler == 'plateau':
           scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=0.1,
                                                               mode="min",
                                                               patience=1,
                                                               verbose=True)
        # TODO FIXME  Untested
        elif args.scheduler == 'swa':
           model = AveragedModel(model).to(DEVICE)
           scheduler = SWALR(
                           optimizer, swa_lr=1e-4, 
                           anneal_epochs=3, 
                           anneal_strategy='cos'
           )   
        
        if args.use_dp:
            if torch.cuda.device_count() > 1:
                print(f"Found {torch.cuda.device_count()} GPUs. Using DataParallel")
                model = nn.DataParallel(model)

        trainer = Trainer(model, optimizer, scheduler, train_loader, valid_loader, DEVICE)

        for epoch in range(1, Config.NB_EPOCHS+1):
            #print(f'Best loss: {best_loss}')

            curr_lr = [group['lr'] for group in optimizer.param_groups][0]

            print(f"\n{'--'*2} EPOCH: {epoch} LR: {curr_lr} {'--'*2}\n")
            params_file = os.path.join(SAVE_MODEL_FOLDER, 'params.txt')
            with open(params_file, 'a+') as f:
               f.write(f'Fold {fold}. Epoch {epoch}: LR {curr_lr}\n')

            if args.enable_wandb:
                # Do this once
                if fold==0 and epoch==0:
                    wandb.watch(trainer.model, log_freq=100)
                wandb.log({"fold": fold, "epoch": epoch})

            # Train for 1 epoch
            trainer.train_one_epoch()
            
            output_model_file = f"{SAVE_MODEL_FOLDER}/bert_model_fold{fold}.bin" #_epoch{epoch}_{trainer.loss_str}.bin"

            # If eval during n steps is not enabled - eval after every epoch
            if not args.eval_steps:
                # Validate for 1 epoch
                current_loss = trainer.valid_one_epoch()

                if current_loss < best_loss:
                    print(f"Saving best model in this fold: {current_loss:.4f}")
                    trainer.loss_str = str(current_loss).replace('.','_')
                    model = trainer.get_model()
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    torch.save(model_to_save.state_dict(), output_model_file)
                    best_loss = current_loss

                print(f"Best RMSE in fold: {fold} was: {best_loss:.4f}")
                print(f"Final RMSE in fold: {fold} was: {current_loss:.4f}")
 
        with open(params_file, 'a+') as f:
            f.write(f"\nBest RMSE in fold: {fold} was: {best_loss:.4f}\n")

        if not args.debug:
            # Load best model and compute OOF predictions
            print('Calculating OOF pred for best model')
            val_pred = get_bert_predictions(valid_data, model_path=output_model_file)
            oof_predictions[valid_idx] = val_pred
     
        del train_set, valid_set, train_loader, valid_loader, model, optimizer, scheduler, trainer
        gc.collect()
        torch.cuda.empty_cache()

    if not args.debug:
        oof_rmse = np.sqrt(mean_squared_error(data['target'], oof_predictions))
        print(f'Our out of folds RMSE is {oof_rmse}')
        with open(params_file, 'a+') as f:
            f.write(f'Our out of folds RMSE is {oof_rmse}')

