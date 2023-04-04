import abc
import re
import logging
from pathlib import Path

import pandas
from pandas import DataFrame
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class BuildFirstEssaySetDataset:
    def __call__(self, source: Path, target: Path) -> None:
        data = self._build(source)
        self._save(data, target)
    
    def _build(self, source: Path) -> DataFrame:
        data = pandas.read_excel(source)
        first_essay_set = data[data["essay_set"] == 1]
        examples = []
        for _, row in first_essay_set.iterrows():
            essay = row["essay"]
            score = row["domain1_score"]
            examples.append({
                "text": essay,
                "score": score
            })
        first_essay_set_dataframe = DataFrame(examples)
        return first_essay_set_dataframe

    def _save(self, data: DataFrame, target: Path) -> None:
        data.to_parquet(target)
        logger.info(f"First essay set dataset was created successfully at {target}")
        logger.info(f"The dataset contains {len(data)} and has the following columns: {', '.join(data.columns)}")


class BaseTextCleaner(abc.ABC):
    @abc.abstractmethod
    def __call__(self, text: str) -> str:
        pass


class TextCleaner(BaseTextCleaner):
    def __call__(self, text: str) -> str:
        uncesored_text = self._remove_censored(text)
        unpuctuated_text = self._remove_punctuation(uncesored_text)
        return unpuctuated_text.lower()

    def _remove_censored(self, text: str) -> str:
        return re.sub(r"@[A-Z]+[0-9]*", "", text)
    
    def _remove_punctuation(self, text: str) -> str:
        return re.sub(r"[^\w\s]", "", text)


class FirstEssaySetDataset(Dataset):
    def __init__(self, path: Path, cleaner: BaseTextCleaner) -> None:
        self.dataset = pandas.read_parquet(path)
        self.cleaner = cleaner

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> tuple[str, float]:
        row = self.dataset.iloc[index]
        text = self.cleaner(row["text"])
        score = row["score"]
        return text, score
