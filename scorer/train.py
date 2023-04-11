from pathlib import Path

from torch.utils.data import DataLoader

from scorer.data import FirstEssaySetDataset, TextCleaner

class TrainScorer:
    def __call__(self, path: Path):
        cleaner = TextCleaner()
        dataset = FirstEssaySetDataset(path, cleaner)
        train_loader = DataLoader(dataset, batch_size=1)

        for sample in train_loader:
            print(sample)

