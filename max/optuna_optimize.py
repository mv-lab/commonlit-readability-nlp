import argparse
import os
import pickle
import optuna
import pandas as pd
from pytorch_lightning.loggers import WandbLogger

import wandb
from optuna import Trial
import pytorch_lightning as pl

parser = argparse.ArgumentParser(description='Process pytorch params.')
parser.add_argument('-model_name', type=str, default='funnel-transformer/large')
args = parser.parse_args()


def objective(trial: Trial):
    from fit_bert_with_mean_predictor import fit, Config

    try:
        with open('key.txt') as f:
            key = f.readline()
        wandb.login(key=key)
    except FileNotFoundError:
        print('Not logging in to wandb - file no found')

    df_train = pd.read_csv('train_folds.csv')
    df_test = pd.read_csv('../input/test.csv')

    config = Config(model_name=args.model_name,
                    batch_size=8,
                    precision=16,
                    accumulate_grad_batches=5,
                    optimizer_name='AdamW',
                    loss_name=trial.suggest_categorical(name='loss_name',
                                                        choices=['rmse_loss', 'rmse_l1_loss']),
                    lr=trial.suggest_float(name='lr', low=2e-6, high=3e-5),
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
                      df_test=df_test)
    if hasattr(return_dict, 'error') and return_dict['error']:
        return 100

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
        logger.experiment.finish()

    return loss


if __name__ == '__main__':
    df_train = pd.read_csv('train_folds.csv')
    df_test = pd.read_csv('../input/test.csv')

    sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
    study = optuna.create_study(sampler=sampler, storage=None, direction='minimize')
    study.enqueue_trial({'lr': 1e-5,
                         'scheduler': 'linear_schedule_with_warmup',
                         "loss_name": 'rmse_loss'})
    study.optimize(objective, n_trials=75, catch=(Exception,))
    with open(f'study_{args.model_name}.pkl', 'wb') as f:
        pickle.dump(study, f)
