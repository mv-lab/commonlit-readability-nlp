import os
from dataclasses import dataclass
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

try:
    with open('key.txt') as f:
        key = f.readline()

        wandb.login(key=key)
except FileNotFoundError:
    print('Not logging in to wandb - file no found')


def rmse_loss(logits, y_true):
    return torch.sqrt(nn.MSELoss()(logits, y_true))


def loss_factory(loss_name):
    if loss_name == 'rmse_loss':
        return rmse_loss
    else:
        raise NotImplementedError


def optimizer_factory(optimizer_name, model, lr):
    if optimizer_name == 'AdamW':
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'word_embeddings.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)  # NOQA

    elif optimizer_name == 'AdamWDifferential':
        # differential learning rate and weight decay
        wd, lr2 = 0.01, 1e-3

        no_decay = ['bias', 'gamma', 'beta']

        group1 = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.']
        group2 = ['layer.4.', 'layer.5.', 'layer.6.', 'layer.7.']
        group3 = ['layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
        group_all = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.', 'layer.6.', 'layer.7.',
                     'layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']

        backbone = model.bert if hasattr(model, 'bert') else model.roberta
        backbone_subset = "bert"
        optimizer_parameters = [
            {'params': [p for n, p in backbone.named_parameters() if
                        not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': wd},
            {'params': [p for n, p in backbone.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],
             'weight_decay_rate': wd, 'lr': lr / 2.6},
            {'params': [p for n, p in backbone.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],
             'weight_decay_rate': wd, 'lr': lr},
            {'params': [p for n, p in backbone.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],
             'weight_decay_rate': wd, 'lr': lr * 2.6},
            {'params': [p for n, p in backbone.named_parameters() if
                        any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': 0.0},
            {'params': [p for n, p in backbone.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],
             'weight_decay_rate': 0.0,
             'lr': lr / 2.6},
            {'params': [p for n, p in backbone.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],
             'weight_decay_rate': 0.0,
             'lr': lr},
            {'params': [p for n, p in backbone.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],
             'weight_decay_rate': 0.0,
             'lr': lr * 2.6},
            {'params': [p for n, p in model.named_parameters() if backbone_subset not in n], 'lr': lr2,
             "momentum": 0.99},
        ]
        return transformers.AdamW(optimizer_parameters, lr=lr)


    else:
        raise NotImplementedError
    return optimizer


def scheduler_factory(scheduler_name_or_scheduler_dict, optimizer, num_training_steps):
    if isinstance(scheduler_name_or_scheduler_dict, dict):
        # custom scheduler
        return scheduler_name_or_scheduler_dict

    if scheduler_name_or_scheduler_dict == 'linear_schedule_with_warmup':
        num_warmup_steps = int(num_training_steps * 0.05)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                 num_warmup_steps=num_warmup_steps,
                                                                 num_training_steps=num_training_steps,
                                                                 )
        scheduler = {'scheduler': scheduler,
                     'interval': 'step',
                     'frequency': 1
                     }

    elif scheduler_name_or_scheduler_dict == 'cosine':
        num_warmup_steps = int(num_training_steps * 0.05)
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=num_warmup_steps,
                                                                 num_training_steps=num_training_steps)
        scheduler = {'scheduler': scheduler,
                     'interval': 'step',
                     'frequency': 1
                     }

    elif scheduler_name_or_scheduler_dict == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=0.1,
                                                               mode="min",
                                                               min_lr=5e-7,
                                                               patience=1,
                                                               verbose=True)
        scheduler = {'scheduler': scheduler,
                     'interval': 'epoch',
                     'reduce_on_plateau': True,
                     'monitor': 'validation_loss'
                     }
    elif scheduler_name_or_scheduler_dict is None:
        scheduler = None
    else:
        raise NotImplementedError
    return scheduler


@dataclass
class Config:
    root_dir: str = '../../lightning_logs'
    seed: int = 0

    lr: float = 2e-5
    optimizer_name: str = 'AdamW'
    loss_name: str = 'rmse_loss'
    scheduler: Optional[str] = None
    epochs: int = 20

    batch_size: int = 16
    accumulate_grad_batches: int = 1
    max_text_length: int = 256
    model_name: str = 'roberta-base'

    text_column_name: str = 'excerpt'
    target_column_name: str = 'target'

    overwrite_train_params: Optional[dict] = None

    def to_str(self):
        return f'lr:{self.lr}_' \
               f'optimizer_name:{self.optimizer_name}_' \
               f'loss_name:{self.loss_name}_' \
               f'scheduler:{self.scheduler}_' \
               f'epochs:{self.epochs}_' \
               f'batch_size:{self.batch_size}_' \
               f'max_text_length:{self.max_text_length}_' \
               f'model_name:{self.model_name}'

    def as_dict(self):
        return {key: getattr(self, key) for key in self.__annotations__.keys()}


class NLPDataset(Dataset):

    def __init__(self, df, tokenized_text):
        self.df = df
        self.tokenized_text = tokenized_text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_dict = self.df.iloc[idx].to_dict()
        for key, value in self.tokenized_text.items():
            assert key not in input_dict
            input_dict[key] = value[idx]
        return input_dict


def as_tensor(input_dict):
    def maybe_to_tensor(x):
        try:
            tensor = torch.from_numpy(np.array(x))
            if len(tensor.shape) == 1:
                tensor = tensor.view(-1, 1)
            if tensor.dtype == torch.float64:
                tensor = tensor.type(torch.float32)  # NOQA
            return tensor
        except TypeError:
            return x

    return {key: maybe_to_tensor(value) for key, value in input_dict.items()}


class TextCollate:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_dicts: List[dict]):
        keys = list(input_dicts[0].keys())
        input_dict = {key: [input_dict[key] for input_dict in input_dicts] for key in keys}
        self.pad(input_dict)
        return as_tensor(input_dict)

    def pad(self, input_dict):
        name2padding = {'input_ids': self.tokenizer.pad_token_id,
                        'attention_mask': 0,
                        'token_type_ids': 0}

        for name, padding_value in name2padding.items():
            if name in input_dict:
                sequences = input_dict[name]
                max_length = max([len(sequence) for sequence in sequences])
                padded_data = np.array(
                    [self._pad_sample(sequence, max_length, padding_value, self.tokenizer.padding_side)
                     for sequence in sequences])
                input_dict[name] = padded_data

    def _pad_sample(self, x, max_length, padding_value, padding_side='right'):
        if padding_side == 'right':
            return x[:max_length] + [padding_value] * (max_length - len(x[:max_length]))
        elif padding_side == 'left':
            return [padding_value] * (max_length - len(x[:max_length])) + x[-max_length:]
        else:
            raise NotImplementedError


class NLPDataModule(pl.LightningDataModule):

    def __init__(self,
                 config: Config,
                 df_train: pd.DataFrame = None,
                 df_valid: pd.DataFrame = None,
                 df_test: pd.DataFrame = None,
                 ):

        super().__init__()
        self.config = config

        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test

        self.batch_size = self.config.batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        self.input_dict_train = None
        self.input_dict_valid = None
        self.input_dict_test = None

    def prepare_data(self, *args, **kwargs):
        self.setup()

    def setup(self, stage=None):
        self.input_dict_train = self.create_input_dict(self.df_train)
        self.input_dict_valid = self.create_input_dict(self.df_valid)
        self.input_dict_test = self.create_input_dict(self.df_test)

    def create_input_dict(self, df):
        if df is None:
            return None

        texts = list(df[self.config.text_column_name].astype(str).fillna("nan").values)
        tokenized_text = self.tokenizer(texts,
                                        add_special_tokens=True,
                                        max_length=self.config.max_text_length,
                                        padding=False,
                                        truncation=True)
        return NLPDataset(df, tokenized_text)

    def train_dataloader(self):
        return self.get_dataloader(mode='train')

    def val_dataloader(self):
        return self.get_dataloader(mode='valid')

    def test_dataloader(self):
        return self.get_dataloader(mode='test')

    def get_dataloader(self, mode):
        input_dict = {'train': self.input_dict_train,
                      'valid': self.input_dict_valid,
                      'test': self.input_dict_test}[mode]
        if input_dict is None:
            return None

        return DataLoader(input_dict,
                          batch_size=self.batch_size,
                          collate_fn=TextCollate(tokenizer=self.tokenizer),
                          num_workers=1,
                          shuffle=True if mode == 'train' else False)


class NLPModel(pl.LightningModule):

    def __init__(self, config: Config, fold=None):
        super().__init__()
        self.config = config
        self.fold = fold

        self.model = self.get_model()
        self.lr = self.config.lr
        self.loss_function = loss_factory(self.config.loss_name)
        self.valid_loss = rmse_loss

    def get_model(self):
        raise NotImplementedError

    def forward(self, input_dict):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = optimizer_factory(self.config.optimizer_name, self.model, self.lr)
        scheduler = self._get_scheduler(self.config.scheduler, optimizer)
        if scheduler:
            return [optimizer], [scheduler]
        return optimizer

    def _get_scheduler(self, scheduler_name, optimizer):
        frequency_optimizer_steps = self.trainer.world_size * self.trainer.accumulate_grad_batches
        num_training_steps = len(self.train_dataloader()) * self.trainer.max_epochs // frequency_optimizer_steps
        return scheduler_factory(scheduler_name, optimizer, num_training_steps)

    def training_step(self, input_dict, batch_num):
        output_dict = self(input_dict)
        loss = self.dict_loss(output_dict)
        self.log('training_loss', loss.detach().cpu())
        if self.fold is not None:
            self.log('fold_id', self.fold)

        return loss

    def validation_step(self, input_dict, batch_nb):
        output_dict = self(input_dict)
        logits = output_dict['logits']
        y_true = input_dict[self.config.target_column_name].type(torch.float32).to(logits.device)
        loss = self.valid_loss(logits, y_true)
        return loss

    def validation_epoch_end(self, outputs):
        loss = torch.stack(outputs).mean()
        print(f'Validation loss after epoch {self.trainer.current_epoch}: {loss}')
        self.log('validation_loss', loss, prog_bar=True)

    def dict_loss(self, input_dict):
        logits = input_dict['logits']
        y_true = input_dict[self.config.target_column_name].type(torch.float32).to(logits.device)
        loss = self.loss_function(logits, y_true)
        return loss

    def get_prediction_df(self, dataloader):
        return_dicts = self.trainer.predict(dataloaders=dataloader)
        keys = return_dicts[0].keys()
        prediction_dict = {}

        for key in keys:
            if key in ['input_ids', 'token_type_ids', 'attention_mask']:
                continue
            elif isinstance(return_dicts[0][key], torch.Tensor) and return_dicts[0][key].shape[-1] != 1:
                # TODO: create one column per dimension
                continue
            elif isinstance(return_dicts[0][key], torch.Tensor) and \
                    return_dicts[0][key].flatten().isnan().type(torch.float).mean().item() != 1:
                x = [list(return_dict[key].cpu().numpy().flatten()) for return_dict in return_dicts]
            else:
                x = [list(return_dict[key]) for return_dict in return_dicts]
            x = [item for sublist in x for item in sublist]
            prediction_dict[key] = x

        return pd.DataFrame(prediction_dict)


class NLPClassificationModel(NLPModel):

    def get_model(self):
        model_config = AutoConfig.from_pretrained(self.config.model_name)
        model_config.num_labels = 1
        model_config.return_dict = True
        return AutoModelForSequenceClassification.from_pretrained(self.config.model_name, config=model_config)

    def forward(self, input_dict):
        assert 'logits' not in input_dict
        token_type_ids = input_dict.get('token_type_ids')
        if token_type_ids is not None:
            logits = self.model(input_ids=input_dict['input_ids'].to(self.device),
                                attention_mask=input_dict['attention_mask'].to(self.device),
                                token_type_ids=token_type_ids.to(self.device)
                                ).logits
        else:
            logits = self.model(input_ids=input_dict['input_ids'].to(self.device),
                                attention_mask=input_dict['attention_mask'].to(self.device),
                                ).logits

        input_dict['logits'] = logits
        return input_dict


def fit(config: Config, df_train, df_test,
        data_module_class=NLPDataModule,
        model_class=NLPClassificationModel,
        overwrite_train_params=None,
        logger_class=WandbLogger
        ):
    dfs_oof = []
    dfs_test_preds = []
    best_weights = []

    logger = None
    if logger_class == WandbLogger:
        logger = logger_class(name=f'mean_pred_{config.model_name}_{config.lr}_{config.scheduler}',
                              project='CommonlitReadabilityTrain',
                              job_type='train')
        logger.log_hyperparams(config.as_dict())
        logger.experiment.save('fit_bert.py')

    for fold in range(df_train['kfold'].max() + 1):
        pl.seed_everything(seed=config.seed)
        datamodule = data_module_class(config=config,
                                       df_train=df_train.loc[df_train['kfold'] != fold],
                                       df_valid=df_train.loc[df_train['kfold'] == fold],
                                       df_test=df_test)
        model = model_class(config=config)
        dirpath = os.path.join(config.root_dir, f'./fold_{fold}')

        checkpoint_callback = ModelCheckpoint(dirpath=dirpath,
                                              filename=config.to_str() + '_{epoch:02d}-{validation_loss:.2f}',
                                              monitor='validation_loss')

        trainer_params = dict(logger=logger,
                              checkpoint_callback=True,
                              callbacks=[checkpoint_callback],
                              gpus='0',
                              accumulate_grad_batches=config.accumulate_grad_batches,
                              default_root_dir=os.path.join(dirpath, config.to_str()),
                              max_epochs=config.epochs,
                              log_every_n_steps=1,
                              min_epochs=1,
                              precision=16,
                              deterministic=True,
                              reload_dataloaders_every_epoch=True)
        if isinstance(overwrite_train_params, dict):
            trainer_params.update(overwrite_train_params)

        trainer = pl.Trainer(**trainer_params)
        trainer.fit(model=model, datamodule=datamodule)
        checkpoint_path = checkpoint_callback.best_model_path
        best_weights.append(checkpoint_path)
        print(f'Loading {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])

        dfs_oof += [model.get_prediction_df(datamodule.val_dataloader())]
        dfs_test_preds += [model.get_prediction_df(datamodule.test_dataloader())]

    df_oof = pd.concat(dfs_oof)
    df_test_preds = pd.concat(dfs_test_preds)

    loss = np.sqrt(np.mean((df_oof[config.target_column_name] - df_oof['logits']) ** 2))
    if isinstance(logger, WandbLogger):
        logger.log_metrics(metrics={'oov_rsme': loss})

    return {'df_oof': df_oof,
            'df_test_preds': df_test_preds,
            'best_weights': best_weights,
            'loss': loss}


if __name__ == '__main__':
    df_train = pd.read_csv('train_folds.csv')
    df_test = pd.read_csv('../../input/test.csv')

    config = Config(model_name='roberta-base',
                    batch_size=4,
                    optimizer_name='AdamWDifferential',
                    lr=1.5e-5,
                    scheduler='plateau',
                    epochs=5,
                    overwrite_train_params={'val_check_interval': 50}
                    )
    overwrite_train_params = config.overwrite_train_params

    pl.seed_everything(seed=config.seed)

    return_dict = fit(config=config,
                      overwrite_train_params=overwrite_train_params,
                      df_train=df_train,
                      df_test=df_test)
    loss = return_dict['loss']
    print(return_dict['loss'])

    df_oof = return_dict['df_oof']
    df_test_preds = return_dict['df_test_preds']

    experiment_name = config.to_str() + f'oof_loss:_{loss}'

    df_oof.to_csv(os.path.join(config.root_dir, 'df_oof_' + experiment_name + '.csv'), index=False)
    df_test_preds.to_csv(os.path.join(config.root_dir, 'df_test_preds_' + experiment_name + '.csv'), index=False)
