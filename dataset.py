from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import json

PAD_IDX = 0


class TinyStoriesDataset(Dataset):
    def __init__(self, files, tokenizer, dataset_size=None, save_to_memory=False, load_from_memory=False, filename=''):
        super().__init__()
        if not load_from_memory:
            corpus = []
            for file in files:
                with open(f"jsons/{file}", 'r') as j:
                    stories = json.loads(j.readline())
                    for story in tqdm(stories):
                        corpus.append(
                            torch.tensor(
                                tokenizer.encode(story["story"], add_eos=True)
                                + [PAD_IDX] * 256,
                                dtype=torch.int16,
                            )[:256][None, :]
                        )
            self.corpus = torch.cat(corpus, dim=0)[:dataset_size, :]
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
