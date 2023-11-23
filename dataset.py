import json
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
import torch

PAD_IDX = 0


class TinyStoriesDataset(Dataset):
    def __init__(self, file, tokenizer, dataset_size=None, save_to_memory=False, load_from_memory=False, filename=''):
        super().__init__()
        if not load_from_memory:
            corpus = []
            with open(file) as j:
                for i in tqdm(range(dataset_size)):
                    string = j.readline()
                    corpus.append(
                        torch.tensor(
                            tokenizer.encode(string, add_bos=True)
                            + [PAD_IDX] * 256,
                            dtype=torch.int16,
                        )[:256][None, :]
                    )
            self.corpus = torch.cat(corpus, dim=0)
            print(self.corpus.shape)
        else:
            self.corpus = torch.load(filename).long()[:dataset_size, :]
            print(self.corpus.dtype)
        if save_to_memory:
            torch.save(self.corpus, filename)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        return self.corpus[index]
