import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np

PAD_IDX = 0


class TinyStoriesDataset(Dataset):
    def __init__(self, jsons, tokenizer, dataset_size=None, save_to_memory=False, load_from_memory=False, filename=''):
        super().__init__()
        if not load_from_memory:
            corpus = []
            for js in jsons:
                with open(js) as j:
                    string = j.readline()

                stories = json.loads(string)
                if dataset_size is not None:
                    stories = stories[:dataset_size]
                for story in tqdm(stories):
                    corpus.append(
                        np.array(
                            tokenizer.encode(story["story"], add_bos=True)
                            + [PAD_IDX] * 256,
                            dtype=np.uint16,
                        )[:256]
                    )
            self.corpus = np.array(corpus, dtype=np.uint16)
        else:
            self.corpus = np.reshape(np.load(filename).astype(np.long), (-1, 256))
        if save_to_memory:
            np.save(filename, self.corpus)
        print(self.corpus)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        return self.corpus[index]
