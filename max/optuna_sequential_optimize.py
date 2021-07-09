import argparse
import os
import pickle
from typing import List

import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
from optuna import Trial, Study
from optuna.distributions import CategoricalDistribution, UniformDistribution, IntUniformDistribution
from optuna.trial import FrozenTrial

import wandb
from fit_bert_with_mean_predictor import fit, Config

try:
    with open('key.txt') as f:
        key = f.readline()
    wandb.login(key=key)
except FileNotFoundError:
    print('Not logging in to wandb - file no found')

parser = argparse.ArgumentParser(description='Process pytorch params.')
parser.add_argument('-model_name', type=str, default='funnel-transformer/large')
args = parser.parse_args()


class RemoveBadWeights:

    def __init__(self, num_models_to_save=5):
        self.num_models_to_save = num_models_to_save

    def __call__(self,
                 study: Study,
                 frozentrial: FrozenTrial):
        trials_to_remove = self.get_trials_to_remove(study)
        for trial_to_remove in trials_to_remove:
            for weight in trial_to_remove.user_attrs.get('best_weights', []):
                try:
                    os.remove(weight)
                except Exception as e:
                    print(f'Could not remove {weight} due to {e}')

    def get_trials_to_remove(self, study) -> List[Trial]:
        completed_trials = [trial for trial in study.trials if trial.value is not None]
        sorted_trials_ids = np.array([trial.number for trial in completed_trials])[np.argsort([trial.value
                                                                                               for trial in
                                                                                               completed_trials])]
        best_id = study.best_trial._trial_id
        assert best_id == sorted_trials_ids[0] or best_id == sorted_trials_ids[-1]
        if best_id == sorted_trials_ids[-1]:
            sorted_trials_ids = sorted_trials_ids[::-1]
        trial_ids_to_remove = sorted_trials_ids[self.num_models_to_save:]
        return [trial for trial in study.trials if trial.number in trial_ids_to_remove]


class NlpTuner:

    def __init__(self):
        self.study = optuna.create_study(storage=None, direction='minimize')

    def run(self) -> None:
        self.tune_learning_rate()
        self.tune_scheduler()
        self.tune_accumulate_grad_batches()
        self.tune_rest()

    def tune_learning_rate(self, n_trials: int = 8) -> None:
        param_name = "lr"
        param_values = np.linspace(5e-6, 5e-4, n_trials).tolist()

        sampler = optuna.samplers.GridSampler({param_name: param_values})
        self.study.sampler = sampler
        self.tune_params([param_name], len(param_values), name='initial_lr_finder')

    def tune_scheduler(self) -> None:
        param_name = "scheduler"
        param_values = ['linear_schedule_with_warmup', 'cosine', 'plateau', None]

        sampler = optuna.samplers.GridSampler({param_name: param_values})
        self.study.sampler = sampler
        self.tune_params([param_name], len(param_values), name='scheduler')

    def tune_accumulate_grad_batches(self) -> None:
        param_name = "accumulate_grad_batches"
        param_values = [1, 2, 5, 10, 20]
        sampler = optuna.samplers.GridSampler({param_name: param_values})
        self.study.sampler = sampler
        self.tune_params([param_name], len(param_values), name='accumulate_grad_batches')

    def tune_rest(self) -> None:
        params = ["optimizer_name", "lr"]
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
        self.study.sampler = sampler
        self.tune_params(params, 20, name='tune_rest')

    def tune_params(self, params, n_trials, name=''):
        possible_tuning_steps = {
            'optimizer_name': (CategoricalDistribution, ['AdamW', 'AdamWNoBias', 'AdamWDifferential']),
            'loss_name': (CategoricalDistribution, ['rmse_loss', 'rmse_l1_loss']),
            'lr': (UniformDistribution, [2e-6, 3e-5]),
            'scheduler': (CategoricalDistribution, ['linear_schedule_with_warmup', 'cosine',
                                                    'plateau', None]),
            'accumulate_grad_batches': (IntUniformDistribution, [1, 10])}
        best_trial = self.study.best_trial
        best_params = dict(optimizer_name='AdamW',
                           loss_name='rmse_loss',
                           lr=2e-5,
                           scheduler='linear_schedule_with_warmup',
                           accumulate_grad_batches=5)
        if best_trial is not None:
            for param in possible_tuning_steps:
                if param:
                    best_params[param] = getattr(best_trial, param)

        def objective(trial: Trial):
            config = Config(model_name=args.model_name,
                            batch_size=best_params['batch_size'],
                            precision=16,
                            accumulate_grad_batches=best_params['accumulate_grad_batches'],
                            optimizer_name=best_params['optimizer_name'],
                            loss_name=best_params['loss_name'],
                            lr=best_params['lr'],
                            epochs=10,
                            scheduler=best_params['scheduler'],
                            overwrite_train_params={'val_check_interval': 0.5}
                            )
            for param in params:
                distribution, values = possible_tuning_steps[param]
                if isinstance(distribution, CategoricalDistribution):
                    suggestion = trial.suggest_categorical(param, values)
                elif isinstance(distribution, UniformDistribution):
                    suggestion = trial.suggest_float(param, low=values[0], high=values[1])
                elif isinstance(distribution, IntUniformDistribution):
                    suggestion = trial.suggest_int(param, low=values[0], high=values[1])
                else:
                    raise NotImplementedError

                setattr(config, param, suggestion)

            return_dict = self.objective(config)
            trial.set_user_attr('best_weights', return_dict['best_weights'])
            return return_dict['loss']

        self.study.optimize(objective, n_trials, callbacks=[RemoveBadWeights(num_models_to_save=4)])
        with open(f'study_{args.model_name}_{name}.pkl', 'wb') as f:
            pickle.dump(self.study, f)

    def objective(self, config: Config):
        df_train = pd.read_csv('train_folds.csv')
        df_test = pd.read_csv('../input/test.csv')

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
        return return_dict


if __name__ == '__main__':
    tuner = NlpTuner()
    try:
        tuner.run()
    except KeyboardInterrupt:
        with open(f'study_{args.model_name}_interruped.pkl', 'wb') as f:
            pickle.dump(tuner.study, f)
