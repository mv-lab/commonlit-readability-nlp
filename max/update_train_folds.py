import pandas as pd

df_train_reduced = pd.read_csv('train_reduced.csv')
df_train_folds = pd.read_csv('train_folds.csv')

df_train_folds = df_train_folds.loc[df_train_folds['id'].isin(df_train_reduced['id'])]
df_train_folds.to_csv('train_folds_reduced.csv', index=False)
