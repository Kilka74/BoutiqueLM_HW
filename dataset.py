import json
from torch.utils.data import Dataset


class TinyStoriesDataset(Dataset):
    def __init__(self, jsons, dataset_size=None):
        super().__init__()
        corpus = []
        for js in jsons:
            with open(js) as j:
                string = j.readline()
            corpus = corpus + json.loads(string)
        if dataset_size is not None:
            self.corpus = corpus[:dataset_size]
        else:
            self.corpus = corpus

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        return self.corpus[index]['story']
