import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from scipy.stats import norm
from tqdm import tqdm

import wandb
from fit_bert import NLPDataModule, NLPClassificationModel, Config

with open('key.txt') as f:
    key = f.readline()

wandb.login(key=key)


class RunningMean:
    def __init__(self, momentum=0.9, epsilon=1e-05):
        self.momentum = momentum
        self.epsilon = epsilon

        self.running_mean = torch.tensor(0)

    def update(self, input):
        mean = input.mean()
        self.running_mean = self.running_mean.to(mean.device)
        self.running_mean = (self.momentum * self.running_mean) + (1.0 - self.momentum) * mean  # .to(input.device)


class NLPClassificationModelFold(NLPClassificationModel):

    def __init__(self, config: Config, fold):
        super().__init__(config)
        self.fold = fold
        self.mu = 0
        self.running_mean = RunningMean()

    def training_step(self, input_dict, batch_num):

        output_dict = self(input_dict)
        self.running_mean.update(output_dict['logits'].detach() - output_dict[self.config.target_column_name].detach())

        output_dict['logits'] = output_dict['logits'] + self.running_mean.running_mean
        loss = self.dict_loss(output_dict)
        self.log('training_loss', loss.detach().cpu())
        self.log('running_mean', self.running_mean.running_mean.cpu())

        if self.fold is not None:
            self.log('fold_id', self.fold)

        return loss

    def calibrate(self):
        dataloder = self.train_dataloader()
        logits = []
        y_true = []
        with torch.no_grad():
            for input_dict in tqdm(dataloder, desc='Calibrating...'):
                output_dict = self(input_dict)
                logits += [output_dict['logits'].cpu().numpy()]
                y_true += [input_dict[self.config.target_column_name].cpu().numpy()]

        mu, sigma = norm.fit(np.concatenate(logits, 0) - np.concatenate(y_true, 0))
        self.mu = mu

    def validation_step(self, input_dict, batch_nb):
        output_dict = self(input_dict)
        logits = output_dict['logits']
        y_true = input_dict[self.config.target_column_name].type(torch.float32).to(logits.device)
        loss = self.valid_loss(logits, y_true)
        return {'loss': loss,
                'y_true': y_true,
                'y_pred': logits}

    def validation_epoch_end(self, outputs):
        y_true = torch.cat([o['y_true'] for o in outputs], 0)
        y_pred = torch.cat([o['y_pred'] for o in outputs], 0)
        self.calibrate()

        loss = self.valid_loss(y_pred, y_true)
        loss_calibrated = self.valid_loss(y_pred - self.mu, y_true)

        self.log('mu', self.mu, prog_bar=True)
        self.log('validation_loss', loss, prog_bar=True)
        self.log('validation_loss_calibrated', loss_calibrated, prog_bar=True)

        print(f'Intercept:{self.mu}')
        print(f'Validation loss after epoch {self.trainer.current_epoch}: {loss}')
        print(f'Validation loss (calibrated) after epoch {self.trainer.current_epoch}: {loss_calibrated}')

    def get_prediction_df(self, dataloader):
        df = super(NLPClassificationModelFold, self).get_prediction_df(dataloader)
        self.calibrate()
        df['mu'] = self.mu


def fit(config: Config, df_train, df_test,
        data_module_class=NLPDataModule,
        model_class=NLPClassificationModelFold,
        overwrite_train_params=None,
        logger_class=WandbLogger
        ):
    dfs_oof = []
    dfs_test_preds = []
    best_weights = []
    logger = None

    if logger_class == WandbLogger:
        logger = logger_class(name=f'calibrate_train_{config.model_name}_{config.lr}_{config.scheduler}',
                              project='CommonlitReadabilityTrain',
                              job_type='train')
        logger.log_hyperparams(config.as_dict())

    for fold in range(df_train['kfold'].max() + 1):
        pl.seed_everything(seed=config.seed)
        datamodule = data_module_class(config=config,
                                       df_train=df_train.loc[df_train['kfold'] != fold],
                                       df_valid=df_train.loc[df_train['kfold'] == fold],
                                       df_test=df_test)

        model = model_class(config=config, fold=fold)
        dirpath = os.path.join(config.root_dir, f'./fold_{fold}')

        checkpoint_callback = ModelCheckpoint(dirpath=dirpath,
                                              filename=config.to_str() + '_{epoch:02d}-{validation_loss:.2f}',
                                              monitor='validation_loss_calibrated')
        callbacks = [checkpoint_callback]
        if logger is not None:
            callbacks += [LearningRateMonitor(logging_interval='step', log_momentum=True)]
        trainer_params = dict(logger=logger,
                              checkpoint_callback=True,
                              callbacks=callbacks,
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
    loss_calibrated = np.sqrt(np.mean((df_oof[config.target_column_name] + df_oof['mu'] - df_oof['logits']) ** 2))

    if isinstance(logger, WandbLogger):
        logger.log_metrics(metrics={'oov_rsme': loss, 'oov_rmse_calibrated': loss_calibrated})
    return {'df_oof': df_oof,
            'df_test_preds': df_test_preds,
            'best_weights': best_weights,
            'loss': loss,
            'loss_calibrated': loss_calibrated}


if __name__ == '__main__':
    df_train = pd.read_csv('train_folds.csv')
    df_test = pd.read_csv('../input/test.csv')

    config = Config(model_name='roberta-base',
                    batch_size=6,
                    optimizer_name='AdamW',
                    loss_name='rmse_loss',
                    scheduler='cosine',
                    lr=2e-5,
                    epochs=10,
                    overwrite_train_params={'val_check_interval': 0.5}
                    )
    overwrite_train_params = config.overwrite_train_params

    pl.seed_everything(seed=config.seed)

    return_dict = fit(config=config,
                      overwrite_train_params=overwrite_train_params,
                      df_train=df_train,
                      df_test=df_test,
                      )
    loss = return_dict['loss']
    print(return_dict['loss'])
    print(return_dict['loss_calibrated'])

    df_oof = return_dict['df_oof']
    df_test_preds = return_dict['df_test_preds']

    experiment_name = config.to_str() + f'oof_loss:_{loss}'

    df_oof.to_csv(os.path.join(config.root_dir, 'df_oof_' + experiment_name + '.csv'), index=False)
    df_test_preds.to_csv(os.path.join(config.root_dir, 'df_test_preds_' + experiment_name + '.csv'), index=False)
