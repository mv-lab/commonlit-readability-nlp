import os
import pickle
import optuna
import pandas as pd
from pytorch_lightning.loggers import WandbLogger

import wandb
from optuna import Trial
from fit_bert_with_mean_predictor import fit, Config
import pytorch_lightning as pl


def objective(trial: Trial):
    df_train = pd.read_csv('../../input/train_folds.csv')
    df_test = pd.read_csv('../../input/test.csv')

    config = Config(lr=trial.suggest_loguniform(name='lr', low=5e-6, high=1e-4),
                    optimizer_name='AdamW',
                    loss_name='rmse_loss',
                    model_name='roberta-base',
                    epochs=5,
                    scheduler=trial.suggest_categorical(name='scheduler',
                                                        choices=['linear_schedule_with_warmup',
                                                                 'cosine',
                                                                 'plateau',
                                                                 None]))
    config = Config(model_name='funnel-transformer/large',
                    batch_size=8,
                    optimizer_name=trial.suggest_categorical(name='optimizer_name',
                                                             choices=['AdamW', 'AdamWDifferential']),
                    loss_name=trial.suggest_categorical(name='loss_name',
                                                        choices=['rmse_loss', 'rmse_l1_loss']),
                    accumulate_grad_batches=4,
                    lr=trial.suggest_loguniform(name='lr', low=5e-6, high=1e-4),
                    epochs=10,
                    scheduler=trial.suggest_categorical(name='scheduler',
                                                        choices=['linear_schedule_with_warmup',
                                                                 'cosine',
                                                                 'plateau',
                                                                 None]),
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

    oof_filepath = os.path.join(config.root_dir, 'df_oof_' + experiment_name + '.csv')
    df_test_filepath = os.path.join(config.root_dir, 'df_test_preds_' + experiment_name + '.csv')

    df_oof.to_csv(oof_filepath, index=False)
    df_test_preds.to_csv(df_test_filepath, index=False)

    logger = return_dict['logger']
    if isinstance(logger, WandbLogger):
        wandb_fn = 'df_oof_' + experiment_name + '.csv'
        df_oof.to_csv(wandb_fn, index=False)
        logger.experiment.save(oof_filepath)
    return loss


if __name__ == '__main__':
    try:
        with open('key.txt') as f:
            key = f.readline()
        wandb.login(key=key)
    except FileNotFoundError:
        print('Not logging in to wandb - file no found')

    df_train = pd.read_csv('train_folds.csv')
    df_test = pd.read_csv('../input/test.csv')

    sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
    study = optuna.create_study(sampler=sampler, storage=None, direction='minimize')
    study.enqueue_trial({'lr': 2e-5, 'scheduler': None})
    study.optimize(objective, n_trials=100, catch=(Exception,))
    with open('study.pkl', 'wb') as f:
        pickle.dump(study, f)