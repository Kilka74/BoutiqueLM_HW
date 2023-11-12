import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

PAD_IDX = 0


class TinyStoriesDataset(Dataset):
    def __init__(self, jsons, tokenizer, dataset_size=None):
        super().__init__()
        corpus = []
        for js in jsons:
            with open(js) as j:
                string = j.readline()

            stories = json.loads(string)
            if dataset_size is not None:
                stories = stories[:dataset_size]
            for story in tqdm(stories):
                corpus = corpus + [
                    torch.tensor(
                        tokenizer.encode(story["story"], add_bos=True)
                        + [PAD_IDX] * 256,
                        dtype=torch.long,
                    )[:256]
                ]
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        return self.corpus[index]
