import argparse
import logging
import os
import pickle
import sys
import time
import warnings
from typing import List

import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
from attr import dataclass
from optuna import Trial, Study, TrialPruned
from optuna.distributions import CategoricalDistribution, UniformDistribution, IntUniformDistribution
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import SuccessiveHalvingPruner
from optuna.trial import FrozenTrial
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.memory import garbage_collection_cuda

import wandb
from fit_bert_with_mean_predictor import fit, Config, FittingError


class PyTorchLightningPruningCallbackLogger(PyTorchLightningPruningCallback):

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            logger = trainer.logger
            if isinstance(logger, WandbLogger):
                logger.experiment.finish()

            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)


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
                    with open(weight + '_removed.txt', 'w') as f:
                        f.write(f'Removed with value {trial_to_remove.value}. \n'
                                f'Best value so far: {study.best_trial.value} \n'
                                f'Best paramers: {study.best_trial.params} \n'
                                f'Stage: {trial_to_remove.user_attrs["stage"]} \n'
                                f'Number of trials: {len(study.trials)}')
                except Exception as e:
                    print(f'Could not remove {weight} due to {e}')

    def get_trials_to_remove(self, study) -> List[Trial]:
        completed_trials = [trial for trial in study.trials if trial.value is not None]
        sorted_trials_ids = np.array([trial.number for trial in completed_trials])[np.argsort([trial.value
                                                                                               for trial in
                                                                                               completed_trials])]
        best_id = study.best_trial._trial_id
        if not (best_id == sorted_trials_ids[0] or best_id == sorted_trials_ids[-1]):
            return []
        if best_id == sorted_trials_ids[-1]:
            sorted_trials_ids = sorted_trials_ids[::-1]
        trial_ids_to_remove = sorted_trials_ids[self.num_models_to_save:]
        return [trial for trial in study.trials if trial.number in trial_ids_to_remove]


@dataclass
class PossibleTuningSteps:
    optimizer_name: tuple = (CategoricalDistribution, ['AdamW', 'AdamWNoBias', 'AdamWNoDecay'])
    loss_name: tuple = (CategoricalDistribution, ['rmse_loss', 'rmse_l1_loss'])
    lr: tuple = (UniformDistribution, [2e-6, 5e-5])
    scheduler: tuple = (CategoricalDistribution, ['linear_schedule_with_warmup', 'cosine',
                                                  'plateau', 'None'])
    accumulate_grad_batches: tuple = (IntUniformDistribution, [1, 10])


class Objective:

    def __init__(self,
                 df_train, params_to_tune, stage,
                 model_name,
                 overwrite_train_params=None,
                 project_name='optuna_optimizing'):
        self.df_train = df_train
        self.params_to_tune = params_to_tune
        self.model_name = model_name
        self.overwrite_train_params = overwrite_train_params
        if isinstance(self.overwrite_train_params, dict) and 'val_check_interval' in self.overwrite_train_params:
            raise AssertionError('val_check_interval must not be overwritten when using '
                                 'Lightning pruning Callback!')

        self.stage = stage
        self.project_name = project_name

    def get_config(self, trial):
        best_params = self.get_best_parameters(trial)
        config = Config(model_name=self.model_name,
                        root_dir='../../optuna_optimize_logs',
                        batch_size=8,
                        precision=16,
                        accumulate_grad_batches=best_params['accumulate_grad_batches'],
                        optimizer_name=best_params['optimizer_name'],
                        loss_name=best_params['loss_name'],
                        lr=best_params['lr'],
                        epochs=10,
                        scheduler=best_params['scheduler'],
                        callbacks={0: [PyTorchLightningPruningCallbackLogger(trial=trial,
                                                                             monitor='validation_loss_calibrated')]},
                        overwrite_train_params=self.overwrite_train_params
                        )
        self.sample_parameters(config, trial)
        return config

    def sample_parameters(self, config, trial):
        # TODO: Move to PossibleTuningSteps
        for parameter in self.params_to_tune:
            distribution, values = getattr(PossibleTuningSteps(), parameter)
            if distribution == CategoricalDistribution:
                suggestion = trial.suggest_categorical(parameter, values)
            elif distribution == UniformDistribution:
                suggestion = trial.suggest_float(parameter, low=values[0], high=values[1])
            elif distribution == IntUniformDistribution:
                suggestion = trial.suggest_int(parameter, low=values[0], high=values[1])
            else:
                raise NotImplementedError(distribution)
            setattr(config, parameter, suggestion)

    def get_best_parameters(self, trial):
        best_params = dict(optimizer_name='AdamW',
                           loss_name='rmse_loss',
                           lr=2e-5,
                           scheduler='linear_schedule_with_warmup',
                           accumulate_grad_batches=5)
        completed_trials = [trial for trial in trial.study.trials if trial.value is not None]
        if len(completed_trials) == 0:
            return best_params

        best_trial = trial.study.best_trial
        if best_trial is not None:
            for parameter in PossibleTuningSteps.__annotations__.keys():
                best_params[parameter] = best_trial.params.get(parameter, best_params[parameter])
        return best_params

    def fit(self, config: Config):
        pl.seed_everything(seed=config.seed)

        return_dict = fit(config=config,
                          df_train=self.df_train,
                          project_name=self.project_name)
        garbage_collection_cuda()

        loss = return_dict.loss
        print(return_dict.loss)
        print(return_dict.loss_calibrated)

        df_oof = return_dict.df_oof
        if isinstance(df_oof, pd.DataFrame):
            experiment_name = config.to_str() + f'oof_loss:_{loss}'
            experiment_name = experiment_name.replace('/', '_')
            oof_filepath = os.path.join(config.root_dir, 'df_oof_' + experiment_name + '.csv')
            os.makedirs(config.root_dir, exist_ok=True)
            df_oof.to_csv(oof_filepath, index=False)
        return return_dict

    def dummy_fit(self, config):
        loss = np.random.random()
        best_weight = f'optuna_testing/{loss}.txt'
        os.makedirs('optuna_testing', exist_ok=True)

        with open(best_weight, 'w') as f:
            f.write(str(config.as_dict()))

        return dict(loss=loss,
                    best_weights=[best_weight])

    def __call__(self, trial: Trial):
        config = self.get_config(trial)
        return_dict = self.fit(config)
        trial.set_user_attr('best_weights', return_dict.best_weights)
        trial.set_user_attr('stage', self.stage)
        return return_dict.loss


class NlpTuner:

    def __init__(self, df_train, model_name, num_trials=-1, time_budget=60 * 60 * 24 * 7, resume=True,
                 project_name='optuna_optimizing', overwrite_train_params=None):
        self.overwrite_train_params = overwrite_train_params
        self.model_name = model_name
        self.project_name = project_name
        self.df_train = df_train
        self.num_trials = num_trials
        self.time_budget = time_budget
        self.resume = resume

        self.completed_trials = 0
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        self.study = optuna.create_study(study_name=model_name + '_' + str(time.time()),
                                         storage=None,
                                         pruner=SuccessiveHalvingPruner(),
                                         direction='minimize')

    def run(self) -> None:
        self.tune_learning_rate()
        self.tune_scheduler()
        self.tune_accumulate_grad_batches()
        self.tune_rest()

        best_trial = self.study.best_trial
        print(f'Best value: {best_trial.value} \n'
              f'Best paramers: {best_trial.params} \n'
              f'Stage: {best_trial.user_attrs["stage"]} \n'
              f'Number of trials: {len(self.study.trials)}')

    def tune_learning_rate(self, n_trials: int = 8) -> None:
        param_name = "lr"
        param_values = np.linspace(5e-6, 5e-5, n_trials).tolist()

        sampler = optuna.samplers.GridSampler({param_name: param_values})
        self.study.sampler = sampler
        self.tune_params([param_name], len(param_values), stage='initial_lr_finder')

    def tune_scheduler(self) -> None:
        param_name = "scheduler"
        param_values = PossibleTuningSteps().scheduler[1]

        sampler = optuna.samplers.GridSampler({param_name: param_values})
        self.study.sampler = sampler
        self.tune_params([param_name], len(param_values), stage='scheduler')

    def tune_accumulate_grad_batches(self) -> None:
        param_name = "accumulate_grad_batches"
        param_values = [10]
        sampler = optuna.samplers.GridSampler({param_name: param_values})
        self.study.sampler = sampler
        self.tune_params([param_name], len(param_values), stage='accumulate_grad_batches')

    def tune_rest(self) -> None:
        params = ["optimizer_name", "lr"]
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
        self.study.sampler = sampler
        self.tune_params(params, 20, stage='tune_rest')

    def tune_params(self, params_to_tune, n_trials, stage=''):
        study_name = f'study_{args.model_name}_{stage}.pkl'.replace('/', '_')
        if self.resume and os.path.exists(study_name):
            with open(study_name, 'rb') as f:
                self.study = pickle.load(f)
                print(f'Loaded study: {study_name}')
            return

        if (self.num_trials > 0 and self.completed_trials < self.num_trials) or self.num_trials < 0:
            objective = Objective(self.df_train, params_to_tune, stage=stage,
                                  model_name=self.model_name,
                                  overwrite_train_params=self.overwrite_train_params,
                                  project_name=self.project_name)
            t_0 = time.time()
            self.study.optimize(objective, n_trials, timeout=self.time_budget,
                                callbacks=[RemoveBadWeights(num_models_to_save=4)],
                                catch=(FittingError, TrialPruned),
                                gc_after_trial=True)
            self.time_budget -= time.time() - t_0
            self.completed_trials += n_trials
        else:
            study_name = 'not_tuned_not_enough_trials_left_' + study_name

        os.makedirs(os.path.dirname(study_name) or './', exist_ok=True)
        with open(study_name, 'wb') as f:
            pickle.dump(self.study, f)


parser = argparse.ArgumentParser(description='Process pytorch params.')
parser.add_argument('-model_name', type=str, default='roberta-base')
args = parser.parse_args()

if __name__ == '__main__':

    try:
        with open('key.txt') as f:
            key = f.readline()
        wandb.login(key=key)
    except FileNotFoundError:
        print('Not logging in to wandb - file no found')

    df_train = pd.read_csv('train_folds.csv')
    tuner = NlpTuner(df_train=df_train,
                     model_name=args.model_name,
                     project_name=f'{args.model_name}_optuna_sequential')
    try:
        tuner.run()
    except KeyboardInterrupt:
        with open(f'study_{args.model_name}_interruped.pkl'.replace('/', '_'), 'wb') as f:
            pickle.dump(tuner.study, f)
