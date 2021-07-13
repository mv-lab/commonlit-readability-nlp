import pandas as pd

from max.optuna_sequential_optimize import NlpTuner

if __name__ == '__main__':
    df_train = pd.read_csv('train_folds.csv')
    df_train = df_train.loc[df_train['kfold'].isin([0, 1])]
    df_train = df_train.sample(200)
    tuner = NlpTuner(df_train=df_train,
                     model_name='distilbert-base-uncased',
                     project_name=None,
                     overwrite_train_params={'max_epochs': 1})
    tuner.run()
