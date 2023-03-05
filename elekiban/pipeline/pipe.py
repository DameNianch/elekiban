from abc import ABCMeta
from abc import abstractmethod
from argparse import ArgumentError
from dataclasses import dataclass
import os
from typing import List
import cv2
import numpy as np

from elekiban.pipeline.pump import AbstractDataPump


def through(x):
    return x


class AbstractPipe(metaclass=ABCMeta):
    def __init__(self, pipe_name: str, data_num: int, adjust_fn, batch_fn) -> None:
        self.pipe_name = pipe_name
        self.data_num = data_num
        self._adjust_fn = adjust_fn
        self._batch_fn = batch_fn

    @abstractmethod
    def generate(self, indices):
        pass

    @abstractmethod
    def _setup(self) -> None:
        pass


@dataclass
class IOPipes:
    inputs: List[AbstractPipe]
    outputs: List[AbstractPipe]


class ImagePipe(AbstractPipe):
    def __init__(self, pipe_name, image_paths, adjust_fn=through, batch_fn=np.array) -> None:
        super().__init__(pipe_name=pipe_name, data_num=len(image_paths), adjust_fn=adjust_fn, batch_fn=batch_fn)
        self._image_paths = image_paths
        self._setup()

    def generate(self, indices):
        return self._batch_fn([self._adjust_fn(cv2.imread(self._image_paths[i])) for i in indices])

    def _setup(self):
        for i_path in self._image_paths:
            # HACK: cv2 does not raise Error...
            # cv2.imread(i_path)
            if not os.path.isfile(i_path):
                raise FileExistsError


class LabelPipe(AbstractPipe):
    def __init__(self, pipe_name, labels, adjust_fn=through, batch_fn=np.array) -> None:
        super().__init__(pipe_name=pipe_name, data_num=len(labels), adjust_fn=adjust_fn, batch_fn=batch_fn)
        self._labels = labels

    def generate(self, indices):
        return self._batch_fn([self._adjust_fn(self._labels[i]) for i in indices])

    def _setup(self):
        pass


class CustomFunctionPipe(AbstractPipe):
    def __init__(self, pipe_name, custom_fn, adjust_fn=through, batch_fn=np.array, data_num=100) -> None:
        super().__init__(pipe_name=pipe_name, data_num=data_num, adjust_fn=adjust_fn, batch_fn=batch_fn)
        # TODO: Rethink the relationship between Pump and CustomFunctionPipe.
        self._custom_fn = custom_fn
        self._setup()

    def generate(self, indices):
        return self._batch_fn([self._adjust_fn(self._custom_fn(i)) for i in indices])

    def _setup(self):
        pass


class PipeWithPump(AbstractPipe):
    def __init__(self, pipe_name: str, data_pump: AbstractDataPump, adjust_fn=through, batch_fn=np.array) -> None:
        super().__init__(pipe_name=pipe_name, data_num=len(data_pump), adjust_fn=adjust_fn, batch_fn=batch_fn)
        self._data_pump = data_pump

    def generate(self, indices):
        return self._batch_fn([self._adjust_fn(self._data_pump[i]) for i in indices])

    def _setup(self):
        pass


class MixedPipe:
    def __init__(self, pipe_name: str, pipes: List[AbstractPipe], weights: List[int], mix_fn=through) -> None:
        # TODO: Consider that MixedPipe is Pipe or not.
        # super().__init__(pipe_name=pipe_name, adjust_fn=adjust_fn, batch_fn=batch_fn)
        self.pipe_name = pipe_name
        self._pipes = pipes
        self._weights = weights
        self._mix_fn = mix_fn
        self._setup()

    def generate(self, indices):
        batch_nums = [len(indices) * i / sum(self._weights) for i in self._weights]
        split_indices = np.split(indices, [int(sum(batch_nums[:i + 1])) for i, _ in enumerate(batch_nums[:-1])])
        return self._mix_fn(np.concatenate([i_pipe.generate(i_inds) for i_pipe, i_inds in zip(self._pipes, split_indices)], axis=0))

    def _setup(self):
        self.data_num = sum([i_pipe.data_num for i_pipe in self._pipes])
        if len(self._pipes) != len(self._weights):
            raise ArgumentError
