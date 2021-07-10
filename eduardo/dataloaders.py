import random
import numpy as np
import pandas as pd

from torch import DataLoader, RandomSampler
from Levenshtein import distance as lev_distance

from .utils import auto_tokenize, tokenize_augment, add_center_to_folds
from .datasets import (TextPairsDataset, CommonLitPairsDataset, CommonLitDataset,
                       AugmentedCommonLitPairsDataset, AugmentedCommonLitDataset, CommonLitBasePairsDataset,
                       CommonLitBaseDataset)


ROOT_PATH = '/content/drive/MyDrive/commonlit'
DATA_PATH = ROOT_PATH + '/data'
SIMPLE_PATH = DATA_PATH + '/simple_wikipedia'


def get_stratified_pairs(target_fold, middle=True, num_bins=8, repeats=6):
    """Input idx: train set filter for folds"""

    n_target_fold = len(target_fold)

    # Matrix of logit differences. Positive value implies doc1 simpler than doc2
    target_mat = np.zeros((n_target_fold, n_target_fold))
    for i, x1 in enumerate(target_fold):
        for j, x2 in enumerate(target_fold):
            target_mat[i][j] = x1 - x2
    # 1/sqrt(2) 1 std of X1-X2 with std(Xi)~0.5
    if middle:
        bool_mat = (np.abs(target_mat) >= 1 / np.sqrt(2)) & (np.abs(target_mat) < 2.5)
    else:
        bool_mat = (np.abs(target_mat) >= 1 / np.sqrt(2)) & (np.abs(target_mat) < 1.5)
        num_bins = 6

    final_indices = []

    for s in range(n_target_fold):
        # All values in row s that have distance between 1/sqrt(2) and 2.5
        vals = target_mat[s][bool_mat[s]]
        # Indices where above is true
        wheres = np.where(bool_mat[s])[0]
        # Bins for the values so we get evenly spaced samples
        bins = np.linspace(min(vals), max(vals), num_bins)
        samples = []
        binned_vals = np.digitize(vals, bins)
        for i in set(binned_vals):
            # Sample 1 from each bin
            samples.append(wheres[np.random.choice(np.where(binned_vals == i)[0])])
        final_indices.append(samples)

    count_indices = np.zeros(len(final_indices))

    all_pairs = []
    # Adjust for bins with too many samples
    for i, samples in enumerate(final_indices):
        for j in samples:
            count_indices[j] += 1
            if count_indices[j] < 20:
                all_pairs.append((i, j))

    if middle:
        # Picking 'repeats' random pairs for every text in the dataset
        others = []
        for _ in range(repeats):
            for i in range(n_target_fold):
                others.append((i, random.randint(0, n_target_fold - 1)))

        all_pairs = all_pairs + others
    # shuffle all samples
    random.shuffle(all_pairs)

    return all_pairs


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """

    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = self.task_name
            yield batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset)
            for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])


def create_simple_wiki_dataloader(split, model_type='roberta-base', batch_size=4, pct_simple=0.5):
    """
    SIMPLE/NORMAL WIKIPEDIA DATASET
    """

    split_dict = {'train': 'training',
                  'valid': 'tuning',
                  'test': 'testing'}

    # Load datasets
    normal_df = pd.read_csv(SIMPLE_PATH + '/normal.{}.txt'.format(split_dict[split]),
                            delimiter="\t",
                            header=None,
                            names=['text'])
    simple_df = pd.read_csv(SIMPLE_PATH + '/simple.{}.txt'.format(split_dict[split]),
                            delimiter="\t",
                            header=None,
                            names=['text'])

    # Normal Wikipedia texts are labeled as 0, Simple Wikipedia texts as 1
    normal_df['label'] = 0
    simple_df['label'] = 1

    # Filter texts so the Levenshtein distance between simple/normal text is at least 60
    # This way, we avoid having comparisons with only minimal changes, like 1 word
    # We also filter by length of text so we don't have very small samples
    distance = np.array([lev_distance(a, b) for a, b in zip(normal_df.text, simple_df.text)])
    filtered_idx = (distance >= 60) & (simple_df.text.apply(lambda x: len(x)) >= 150)
    normal_df = normal_df[filtered_idx].reset_index(drop=True)
    simple_df = simple_df[filtered_idx].reset_index(drop=True)

    # Tokenize data
    normal_input_ids, normal_attention_masks, normal_labels = auto_tokenize(normal_df.text, normal_df.label, model_type)
    simple_input_ids, simple_attention_masks, simple_labels = auto_tokenize(simple_df.text, simple_df.label, model_type)
    # Create PyTorch dataset
    simple_dataset = TextPairsDataset(simple_input_ids, simple_attention_masks, simple_labels,
                                      normal_input_ids, normal_attention_masks, normal_labels)

    simple_sampler = RandomSampler(simple_dataset,
                                   replacement=True,
                                   num_samples=int(pct_simple * len(simple_dataset)))
    # Create dataloader
    simple_dataloader = DataLoader(simple_dataset, batch_size=batch_size, sampler=simple_sampler)

    return simple_dataloader


def create_books_dataloader(frac=1, batch_size=4, model_type='roberta-base'):
    """
    BOOKS DATASET
    """

    # Load data
    books_df = pd.read_csv(DATA_PATH + '/book_pairs_baseline.csv').sample(frac=frac)
    books_df.Easy = books_df.Easy.str.replace("\n", ' ')
    books_df.Hard = books_df.Hard.str.replace("\n", ' ')

    # Tokenize data
    easy_input_ids, easy_attention_masks, easy_labels = auto_tokenize(books_df.Easy,
                                                                      np.ones(len(books_df)),
                                                                      model_type)
    hard_input_ids, hard_attention_masks, hard_labels = auto_tokenize(books_df.Hard,
                                                                      np.zeros(len(books_df)),
                                                                      model_type)
    # Create PyTorch dataset
    books_dataset = TextPairsDataset(easy_input_ids, easy_attention_masks, easy_labels,
                                     hard_input_ids, hard_attention_masks, hard_labels)
    books_dataloader = DataLoader(books_dataset, shuffle=True, batch_size=batch_size)

    return books_dataloader


def create_science_dataloader(frac=1, batch_size=4, model_type='roberta-base'):
    """
    SCIENCE DATASET
    """

    # Load data
    science_df = pd.read_csv(DATA_PATH + '/science_pairs_baseline.csv').sample(frac=frac)
    science_df.Easy = science_df.Easy.str.replace("\n", ' ')
    science_df.Hard = science_df.Hard.str.replace("\n", ' ')

    easy_input_ids, easy_attention_masks, easy_labels = auto_tokenize(science_df.Easy,
                                                                      np.ones(len(science_df)),
                                                                      model_type)
    hard_input_ids, hard_attention_masks, hard_labels = auto_tokenize(science_df.Hard,
                                                                      np.zeros(len(science_df)),
                                                                      model_type)
    # Create PyTorch dataset
    science_dataset = TextPairsDataset(easy_input_ids, easy_attention_masks, easy_labels,
                                       hard_input_ids, hard_attention_masks, hard_labels)
    science_dataloader = DataLoader(science_dataset, shuffle=True, batch_size=batch_size)

    return science_dataloader


def generate_dataloaders(texts,
                         raw_labels,
                         train_idx,
                         valid_idx,
                         batch_size=4,
                         tokenizer_name='roberta-base',
                         num_bins=8,
                         repeats=6,
                         pct_pairs=0.4,
                         pct_pairs_split=0.4,
                         pct_simple=0.5,
                         frac_science=1,
                         frac_books=1,
                         num_base_samples=200,
                         augment=False):
    aug_df = pd.read_csv(DATA_PATH + '/train_augmented_kfold1.csv')
    aug_df = add_center_to_folds(aug_df)
    aug_input_ids, aug_attention_masks, _ = tokenize_augment(aug_df.loc[train_idx], tokenizer_name)

    # Tokenize data
    input_ids, attention_masks, labels = auto_tokenize(texts, raw_labels, tokenizer_name)

    ###### Create CommonLit datasets #######

    # Train Fold Datasets

    stratified_pairs = get_stratified_pairs(raw_labels[train_idx],
                                            middle=True,
                                            num_bins=num_bins,
                                            repeats=repeats)

    stratified_pairs_split = get_stratified_pairs(raw_labels[train_idx],
                                                  middle=False,
                                                  num_bins=num_bins,
                                                  repeats=repeats)

    commonlit_pairs_dataset = CommonLitPairsDataset(input_ids[train_idx],
                                                    attention_masks[train_idx],
                                                    labels[train_idx],
                                                    stratified_pairs)

    if augment:

        commonlit_pairs_split_dataset = AugmentedCommonLitPairsDataset(aug_input_ids,
                                                                       aug_attention_masks,
                                                                       labels[train_idx],
                                                                       stratified_pairs_split
                                                                       )

        augmented_commonlit_dataset = AugmentedCommonLitDataset(aug_input_ids,
                                                                aug_attention_masks,
                                                                labels[train_idx].float())

    else:

        commonlit_pairs_split_dataset = CommonLitPairsDataset(input_ids[train_idx],
                                                              attention_masks[train_idx],
                                                              labels[train_idx],
                                                              stratified_pairs_split
                                                              )

    commonlit_base_pairs_dataset = CommonLitBasePairsDataset(input_ids[train_idx],
                                                             attention_masks[train_idx],
                                                             labels[train_idx].float())

    commonlit_base_dataset = CommonLitBaseDataset(input_ids[train_idx],
                                                  attention_masks[train_idx],
                                                  labels[train_idx].float(),
                                                  num_base_samples=num_base_samples)

    train_commonlit_dataset = CommonLitDataset(input_ids[train_idx],
                                               attention_masks[train_idx],
                                               labels[train_idx].float())

    # Valid Fold Datasets
    valid_commonlit_dataset = CommonLitDataset(input_ids[valid_idx],
                                               attention_masks[valid_idx],
                                               labels[valid_idx].float())

    ###### Create dataloaders ######

    # Train Fold Dataloaders

    pairs_sampler = RandomSampler(commonlit_pairs_dataset,
                                  replacement=True,
                                  num_samples=int(pct_pairs * len(commonlit_pairs_dataset)))

    commonlit_pairs_dataloader = DataLoader(commonlit_pairs_dataset,
                                            sampler=pairs_sampler,
                                            batch_size=batch_size)

    pairs_split_sampler = RandomSampler(commonlit_pairs_split_dataset,
                                        replacement=True,
                                        num_samples=int(pct_pairs_split * len(commonlit_pairs_split_dataset)))

    commonlit_pairs_split_dataloader = DataLoader(commonlit_pairs_split_dataset,
                                                  sampler=pairs_split_sampler,
                                                  batch_size=batch_size)

    train_commonlit_dataloader = DataLoader(train_commonlit_dataset,
                                            shuffle=True,
                                            batch_size=batch_size)

    commonlit_base_pairs_dataloader = DataLoader(commonlit_base_pairs_dataset,
                                                 shuffle=True,
                                                 batch_size=batch_size)

    commonlit_base_dataloader = DataLoader(commonlit_base_dataset,
                                           shuffle=True,
                                           batch_size=1)

    books_dataloader = create_books_dataloader(frac_books,
                                               batch_size=batch_size,
                                               model_type=tokenizer_name)

    science_dataloader = create_science_dataloader(frac_science,
                                                   batch_size=batch_size,
                                                   model_type=tokenizer_name)

    simple_dataloader = create_simple_wiki_dataloader('train',
                                                      batch_size=batch_size,
                                                      model_type=tokenizer_name,
                                                      pct_simple=pct_simple)

    if augment:
        augmented_commonlit_dataloader = DataLoader(augmented_commonlit_dataset,
                                                    shuffle=True,
                                                    batch_size=batch_size)

    # Valid Fold Dataloaders
    valid_commonlit_dataloader = DataLoader(valid_commonlit_dataset,
                                            shuffle=False,
                                            batch_size=batch_size)

    ###### Create MultiTask Dataloaders ######

    train_dataloader_dict = {'books': DataLoaderWithTaskname('books', books_dataloader),
                             'science': DataLoaderWithTaskname('science', books_dataloader),
                             'simple': DataLoaderWithTaskname('simple', simple_dataloader),
                             'commonlit': DataLoaderWithTaskname('commonlit', train_commonlit_dataloader),
                             'commonlit_pairs': DataLoaderWithTaskname('commonlit_pairs', commonlit_pairs_dataloader),
                             # 'commonlit_base': DataLoaderWithTaskname('commonlit_base', commonlit_base_dataloader),
                             # 'commonlit_base_pairs': DataLoaderWithTaskname('commonlit_base_pairs', commonlit_base_pairs_dataloader),
                             'commonlit_pairs_split': DataLoaderWithTaskname('commonlit_pairs_split',
                                                                             commonlit_pairs_split_dataloader)
                             }

    if augment:
        train_dataloader_dict['augmented_commonlit'] = DataLoaderWithTaskname('augmented_commonlit',
                                                                              augmented_commonlit_dataloader)

    valid_dataloader_dict = {'commonlit': DataLoaderWithTaskname('commonlit', valid_commonlit_dataloader)}

    train_dataloader = MultitaskDataloader(train_dataloader_dict)
    valid_dataloader = MultitaskDataloader(valid_dataloader_dict)

    return train_dataloader, valid_dataloader

