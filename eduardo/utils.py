import torch
from transformers import AutoTokenizer


def add_center_to_folds(train):
    """
    Add the sample with target 0 to all the folds
    """

    n_train = len(train)
    pivot = train[train.standard_error==0].index[0]

    for _ in range(5):
        train.loc[len(train)] = train.loc[pivot].values

    train.loc[n_train:n_train+4, 'kfold'] =  [0,1,2,3,4]

    train = train.drop_duplicates().reset_index(drop=True)

    return train


def auto_tokenize(data, targets, model_type='roberta-base'):

    if not isinstance(data, list):
        data = data.to_list()

    tokenizer = AutoTokenizer.from_pretrained(model_type)

    encoded_input = tokenizer(
                              data,
                              padding='max_length',
                              truncation=True,
                              max_length=300,
                              return_tensors='pt',
                              )

    input_ids = encoded_input['input_ids']
    attention_masks = encoded_input['attention_mask']
    labels = torch.tensor(targets).resize_(len(targets),1)

    return input_ids, attention_masks, labels


def tokenize_augment(aug_df, model_type):

    aug_input_ids = []
    aug_attention_masks = []

    for aug_col in ['aug0', 'aug1', 'aug2', 'aug3', 'aug4']:

        input_ids, attention_masks, aug_labels = auto_tokenize(aug_df[aug_col], aug_df.target.to_list(), model_type=model_type)
        aug_input_ids.append(input_ids)
        aug_attention_masks.append(attention_masks)

    return aug_input_ids, aug_attention_masks, aug_labels

