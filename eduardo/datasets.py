from torch.utils.data import Dataset


class CommonLitDataset(Dataset):
    """CommonLit dataset."""

    def __init__(self, input_ids, attention_masks, labels):

        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.n = len(input_ids)

    def __len__(self):

        return self.n

    def __getitem__(self, idx):

        sample = {'input_ids': self.input_ids[idx],
                  'attention_masks': self.attention_masks[idx],
                  'labels': self.labels[idx],
                }

        return sample


class AugmentedCommonLitDataset(Dataset):
    """Augmented CommonLit dataset."""

    def __init__(self, input_ids, attention_masks, labels):

        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.n = len(labels)
        # input ids is a different size from labels, includes all augments
        self.n_samples = len(input_ids)

    def __len__(self):

        return self.n

    def __getitem__(self, idx):

        aug_idx = random.randint(0, self.n_samples-1)

        sample = {'input_ids': self.input_ids[aug_idx][idx],
                  'attention_masks': self.attention_masks[aug_idx][idx],
                  'labels': self.labels[idx],
                 }

        return sample


class CommonLitPairsDataset(Dataset):
    """CommonLit Pairs dataset."""

    def __init__(self, input_ids, attention_masks, labels, pairs):

        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.pairs = pairs
        self.n = len(pairs)

    def __len__(self):

        return self.n

    def __getitem__(self, idx):

        idx1 = self.pairs[idx][0]
        idx2 = self.pairs[idx][1]

        sample = {
                  'text1':{'input_ids': self.input_ids[idx1],
                  'attention_masks': self.attention_masks[idx1],
                  'labels': self.labels[idx1],
                  },
                  'text2':{'input_ids': self.input_ids[idx2],
                  'attention_masks': self.attention_masks[idx2],
                  'labels': self.labels[idx2],
                  },
                 }

        return sample


class AugmentedCommonLitPairsDataset(Dataset):
    """Augmented CommonLit Pairs dataset."""

    def __init__(self, input_ids, attention_masks, labels, pairs):

        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.pairs = pairs
        self.n = len(pairs)
        # input ids is a different size from labels, includes all augments
        self.n_samples = len(input_ids)

    def __len__(self):

        return self.n

    def __getitem__(self, idx):

        idx1 = self.pairs[idx][0]
        idx2 = self.pairs[idx][1]

        aug_idx1 = random.randint(0, self.n_samples-1)
        aug_idx2 = random.randint(0, self.n_samples-1)

        sample = {
                  'text1':{'input_ids': self.input_ids[aug_idx1][idx1],
                  'attention_masks': self.attention_masks[aug_idx1][idx1],
                  'labels': self.labels[idx1],
                  },
                  'text2':{'input_ids': self.input_ids[aug_idx2][idx2],
                  'attention_masks': self.attention_masks[aug_idx2][idx2],
                  'labels': self.labels[idx2],
                  },
                 }

        return sample


class TextPairsDataset(Dataset):
    """TextPairsDataset dataset."""

    def __init__(self, input_ids_1, attention_masks_1, labels_1,
                       input_ids_2, attention_masks_2, labels_2):

        self.input_ids_1 = input_ids_1
        self.attention_masks_1 = attention_masks_1
        self.labels_1 = labels_1
        self.input_ids_2 = input_ids_2
        self.attention_masks_2 = attention_masks_2
        self.labels_2 = labels_2
        self.n = len(labels_1)

    def __len__(self):

        return self.n

    def __getitem__(self, idx):

        sample = {
                  'text1':{'input_ids': self.input_ids_1[idx],
                  'attention_masks': self.attention_masks_1[idx],
                  'labels': self.labels_1[idx],
                  },
                  'text2':{'input_ids': self.input_ids_2[idx],
                  'attention_masks': self.attention_masks_2[idx],
                  'labels': self.labels_2[idx],
                  },
                 }

        return sample


class AugmentedTextPairsDataset(Dataset):
    """Augmented TextPairsDataset dataset."""

    def __init__(self, input_ids_1, attention_masks_1, labels_1,
                       input_ids_2, attention_masks_2, labels_2):

        self.input_ids_1 = input_ids_1
        self.attention_masks_1 = attention_masks_1
        self.labels_1 = labels_1
        self.input_ids_2 = input_ids_2
        self.attention_masks_2 = attention_masks_2
        self.labels_2 = labels_2
        self.n = len(labels_1)
        # input ids is a different size from labels, includes all augments
        self.n_samples = len(input_ids_1)

    def __len__(self):

        return self.n

    def __getitem__(self, idx):

        aug_idx1 = random.randint(0, self.n_samples-1)
        aug_idx2 = random.randint(0, self.n_samples-1)

        sample = {
                  'text1':{'input_ids': self.input_ids_1[aug_idx1][idx],
                  'attention_masks': self.attention_masks_1[aug_idx1][idx],
                  'labels': self.labels_1[idx],
                  },
                  'text2':{'input_ids': self.input_ids_2[aug_idx2][idx],
                  'attention_masks': self.attention_masks_2[aug_idx2][idx],
                  'labels': self.labels_2[idx],
                  },
                 }

        return sample


class CommonLitBaseDataset(Dataset):
    """CommonLit dataset."""

    def __init__(self, input_ids, attention_masks, labels, num_base_samples=200):

        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.n = len(input_ids)
        self.center = np.where(labels==0)[0][0]
        self.num_base_samples = num_base_samples

    def __len__(self):

        return self.num_base_samples

    def __getitem__(self, idx):

        sample = {'input_ids': self.input_ids[self.center],
                  'attention_masks': self.attention_masks[self.center],
                  'labels': self.labels[self.center],
                }

        return sample


class CommonLitBasePairsDataset(Dataset):
    """CommonLit dataset."""

    def __init__(self, input_ids, attention_masks, labels):

        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.n = len(input_ids)
        self.center = np.where(labels==0)[0][0]

    def __len__(self):

        return self.n

    def __getitem__(self, idx):

        if self.labels[idx] < 0:

            sample = {
                      'text1':{'input_ids': self.input_ids[self.center],
                      'attention_masks': self.attention_masks[self.center],
                      'labels': self.labels[self.center],
                      },
                      'text2':{'input_ids': self.input_ids[idx],
                      'attention_masks': self.attention_masks[idx],
                      'labels': self.labels[idx],
                      },
                    }

        else:

            sample = {
                      'text1':{'input_ids': self.input_ids[idx],
                      'attention_masks': self.attention_masks[idx],
                      'labels': self.labels[idx],
                      },
                      'text2':{'input_ids': self.input_ids[self.center],
                      'attention_masks': self.attention_masks[self.center],
                      'labels': self.labels[self.center],
                      },
                    }


        return sample
