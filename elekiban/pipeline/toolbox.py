from abc import ABCMeta, abstractmethod
import random
from typing import List
from xml.dom import ValidationErr

from elekiban.pipeline.pipe import AbstractPipe, IOPipes


def through(inputs, outputs):
    return [inputs, outputs]


class AbstractFaucet(metaclass=ABCMeta):
    @abstractmethod
    def _setup(self):
        pass

    @abstractmethod
    def _turn_on(self, indices):
        pass

    @abstractmethod
    def turn_on(self, indices):
        pass

    @abstractmethod
    def get_output_names(self, indices):
        pass


class SimpleFaucet:
    def __init__(self, input_pipelines: List[AbstractPipe], output_pipelines: List[AbstractPipe], batch_size: int, pairing_adjust_fn=through) -> None:
        self._iopipes = IOPipes(inputs=input_pipelines, outputs=output_pipelines)
        self.batch_size = batch_size
        self._pairing_adjust_fn = pairing_adjust_fn
        self._setup()

    def _setup(self):
        data_nums = {}
        for i_pipeline in self._iopipes.inputs + self._iopipes.outputs:
            data_nums[i_pipeline.pipe_name] = i_pipeline.data_num

        if len(set(data_nums.values())) != 1:
            raise ValidationErr(f"dataset nums should be equal for all dataset. {data_nums}")
        else:
            print(f"dataset num are same for all dataset.{data_nums}")
            data_num = list(set(data_nums.values()))[0]
            self.iteration = int(data_num / self.batch_size)
            self._indices = list(range(data_num))
            pass

    def _get_inputs(self, remained_indices: List[int]):
        return {p.pipe_name: p.generate(remained_indices[: self.batch_size]) for p in self._iopipes.inputs}

    def _get_outputs(self, remained_indices: List[int]):
        return {p.pipe_name: p.generate(remained_indices[: self.batch_size]) for p in self._iopipes.outputs}

    def turn_on(self) -> dict:
        while True:
            random.shuffle(self._indices)
            remained_indices = self._indices
            for _ in range(self.iteration):
                yield self._pairing_adjust_fn(self._get_inputs(remained_indices), self._get_outputs(remained_indices))
                remained_indices = remained_indices[: self.batch_size]

    def get_output_names(self):
        return [i_pipeline.pipe_name for i_pipeline in self._iopipes.outputs]
