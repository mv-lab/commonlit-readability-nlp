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
