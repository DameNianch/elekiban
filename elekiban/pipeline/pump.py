from abc import ABC, abstractmethod
import csv
import json
from os import PathLike
from typing import List
import cv2
import numpy as np


def through(x):
    return x


def load_csv(csv_path) -> list:
    lables = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for i_row in reader:
            lables.append(i_row)
    return np.array(lables).astype(float)


def load_jsonl(jsonl_path) -> List[dict]:
    with open(jsonl_path, "r") as f:
        return [json.loads(j) for j in list(f)]


class AbstractDataPump(ABC):
    def __init__(self, listed_data=None, data_path: PathLike = None, **kwargs):
        if listed_data is not None:
            self._listed_data = listed_data
        else:
            self._listed_data = self._read_fn(data_path)
        self._kwargs = kwargs

    def __len__(self) -> int:
        return len(self._listed_data)

    @abstractmethod
    def __getitem__(self, key: int):
        pass


# class ImagePump(AbstractDataPump):
#     def __getitem__(self, key) -> np.ndarray:
#         return cv2.imread(self._listed_data[key])


# class TextLinePump(AbstractDataPump):
#     def __getitem__(self, key):
#         return self._listed_data[key]


class LabelPump(AbstractDataPump):
    def _read_fn(self, csv_path) -> list:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            labels = [i_row for i_row in reader]
        return np.array(labels).astype(float)

    def __getitem__(self, key) -> np.ndarray:
        return np.array(self._listed_data[key])


class FixedLengthWavePump(AbstractDataPump):
    def _read_fn(self, data_path: PathLike) -> list:
        with open(data_path, "r") as f:
            return [json.loads(j) for j in list(f)]

    def __getitem__(self, key) -> np.ndarray:
        return np.array([self._listed_data[key][x] for x in self._kwargs["axes"]]).transpose(1, 0)
