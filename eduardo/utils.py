import torch
from transformers import AutoTokenizer


def add_center_to_folds(train):
    """
    Add the sample with target 0 to all the folds
    """

    n_train = len(train)
    pivot = train[train.standard_error == 0].index[0]

    for _ in range(5):
        train.loc[len(train)] = train.loc[pivot].values

    train.loc[n_train:n_train+4, 'kfold'] = [0, 1, 2, 3, 4]

    train = train.drop_duplicates().reset_index(drop=True)

    return train


def auto_tokenize(data, targets, model_type='roberta-large'):

    if not isinstance(data, list):
        data = data.to_list()

    tokenizer = AutoTokenizer.from_pretrained(model_type)

    encoded_input = tokenizer(
                              data,
                              padding='max_length',
                              truncation=True,
                              max_length=256,
                              return_tensors='pt',
                              )

    input_ids = encoded_input['input_ids']
    attention_masks = encoded_input['attention_mask']
    labels = torch.tensor(targets).resize_(len(targets), 1)

    return input_ids, attention_masks, labels
