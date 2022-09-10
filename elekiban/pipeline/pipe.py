from abc import ABCMeta
from abc import abstractmethod
from argparse import ArgumentError
import cv2
import numpy as np

from elekiban.pipeline.pump import AbstractDataPump


def through(x):
    return x


class AbstractPipe(metaclass=ABCMeta):
    @abstractmethod
    def generate(self, indices):
        pass

    @abstractmethod
    def _setup(self) -> None:
        pass


class ImagePipe(AbstractPipe):
    def __init__(self, pipe_name, image_paths, adjust_fn=through, batch_fn=through) -> None:
        self.pipe_name = pipe_name
        self._image_paths = image_paths
        self._adjust_fn = adjust_fn
        self._batch_fn = batch_fn
        self._setup()
        self.data_num = len(image_paths)

    def generate(self, indices):
        return self._batch_fn(np.array([self._adjust_fn(cv2.imread(self._image_paths[i])) for i in indices]))

    def _setup(self):
        for i_path in self._image_paths:
            try:
                cv2.imread(i_path)
            except BaseException:
                print(f"Cannot open {i_path}")
        print("ok")


class LabelPipe(AbstractPipe):
    def __init__(self, pipe_name, labels, adjust_fn=through, batch_fn=through) -> None:
        self.pipe_name = pipe_name
        self._labels = labels
        self._adjust_fn = adjust_fn
        self._batch_fn = batch_fn
        self.data_num = len(labels)

    def generate(self, indices):
        return self._batch_fn(np.array([self._adjust_fn(self._labels[i]) for i in indices]))

    def _setup(self):
        pass


class CustomFunctionPipe(AbstractPipe):
    def __init__(self, pipe_name, custom_fn, adjust_fn=through, batch_fn=through, data_num=100) -> None:
        self._pipe_name = pipe_name
        self._custom_fn = custom_fn
        self._adjust_fn = adjust_fn
        self._batch_fn = batch_fn
        self.data_num = data_num
        self._setup()

    def generate(self, indices):
        return self._batch_fn(np.array([self._adjust_fn(self._custom_fn(i)) for i in indices]))

    def _setup(self):
        pass


class PipeWithPump(AbstractPipe):
    def __init__(self, pipe_name: str, data_pump: AbstractDataPump, adjust_fn=through, batch_fn=through) -> None:
        self.pipe_name = pipe_name
        self._data_pump = data_pump
        self._adjust_fn = adjust_fn
        self._batch_fn = batch_fn
        self.data_num = len(data_pump)

    def generate(self, indices):
        return self._batch_fn(np.array([self._adjust_fn(self._data_pump[i]) for i in indices]))

    def _setup(self):
        pass


class MixedPipe(AbstractPipe):
    def __init__(self, pipe_name: str, pipes: list, weights: list, mix_fn=through) -> None:
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
