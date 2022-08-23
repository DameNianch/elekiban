from argparse import ArgumentError
from dataclasses import dataclass
import os
from typing import List

import tensorflow as tf

from ..pipeline.toolbox import AbstractFaucet


def train(model: tf.keras.Model,
          train_faucet: AbstractFaucet,
          valid_faucet: AbstractFaucet,
          epochs: int,
          optimizer=tf.keras.optimizers.Adam(),
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=tf.keras.losses.MeanSquaredError(),
          loss_weights=None,
          output_path="./") -> os.PathLike:

    def allocate_fn(fn):
        if not isinstance(fn, dict):
            return {i_key: fn for i_key in train_faucet.get_output_names()}

    loss = allocate_fn(loss)
    metrics = allocate_fn(metrics)
    save_model = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_path, "model.h5"),
                                                    monitor="val_loss",
                                                    verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=False,
                                                    mode="auto")

    try:
        model.summary()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)
        model.fit(
            x=train_faucet.turn_on(), batch_size=train_faucet.batch_size, epochs=epochs, steps_per_epoch=train_faucet.iteration,
            validation_data=valid_faucet.turn_on(), validation_batch_size=valid_faucet.batch_size, validation_steps=valid_faucet.iteration,
            callbacks=[save_model]
        )
        return output_path
    except Exception as e:
        print("[*]Fail to training")
        print(e)


@dataclass
class TrainingBlock:
    model_chain: List[tf.keras.Model]
    train_faucet: AbstractFaucet
    valid_faucet: AbstractFaucet = None
    fixed_model_ids: List[int] = None


@dataclass
class TrainingRule:
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MeanSquaredError()
    metrics = tf.keras.losses.MeanSquaredError()
    loss_weights: List[int] = None
    epoch: int = None


def sub_train(block: TrainingBlock, rule: TrainingRule, output_dir="./"):
    # HACK _allocate_fnは雑すぎる
    def _allocate_fn(fn):
        if not isinstance(fn, dict):
            return {i_key: fn for i_key in block.train_faucet.get_output_names()}
        else:
            return fn

    def _join_model(model_chain, fixed_model_ids) -> tf.keras.Model:
        for i in fixed_model_ids:
            model_chain[i].trainable = False
        joined_model = model_chain[0]
        for i_model in model_chain[1:]:
            joined_model = tf.keras.Model(inputs=joined_model.input, outputs=i_model.output)
        return joined_model

    rule.loss = _allocate_fn(rule.loss)
    rule.metrics = _allocate_fn(rule.metrics)
    joined_model = _join_model(block.model_chain, block.fixed_model_ids)
    joined_model.summary()
    joined_model.compile(optimizer=rule.optimizer, loss=rule.loss, metrics=rule.metrics, loss_weights=rule.loss_weights)
    fit_callbacks = [tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, "model.h5"),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto")]
    try:
        joined_model.fit(x=block.train_faucet.turn_on(),
                         batch_size=block.train_faucet.batch_size,
                         epochs=rule.epoch,
                         steps_per_epoch=block.train_faucet.iteration,
                         validation_data=block.valid_faucet.turn_on(),
                         validation_batch_size=block.valid_faucet.batch_size,  # valid_faucet=Noneのときにやばい。
                         validation_steps=block.valid_faucet.iteration,
                         callbacks=fit_callbacks)
    except Exception as e:
        print("[*]Fail to training")
        print(e)


class Trainer:
    def __init__(self,
                 training_block: TrainingBlock,
                 epochs: int,
                 optimizer=tf.keras.optimizers.Adam(),
                 loss=tf.keras.losses.MeanSquaredError(),
                 metrics=tf.keras.losses.MeanSquaredError(),
                 loss_weights=None,
                 output_path="./") -> None:

        self._training_block = training_block
        self._epochs = epochs
        self._setup(optimizer, loss, metrics, loss_weights, output_path)

    def _allocate_fn(self, fn):
        if not isinstance(fn, dict):
            return {i_key: fn for i_key in self.train_faucet.get_output_names()}

    def _setup(self, optimizer, loss, metrics, loss_weights, output_path):
        self._loss = self._allocate_fn(loss)
        self._metrics = self._allocate_fn(metrics)
        self._save_model = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_path, "model.h5"),
                                                              monitor="val_loss",
                                                              verbose=1,
                                                              save_best_only=True,
                                                              save_weights_only=False,
                                                              mode="auto")
        self.model.summary()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)

    def __call__(self):
        try:
            self.model.fit(
                x=self._train_faucet.turn_on(), batch_size=self._train_faucet.batch_size, epochs=self._epochs, steps_per_epoch=self._train_faucet.iteration,
                validation_data=self._valid_faucet.turn_on(), validation_batch_size=self._valid_faucet.batch_size, validation_steps=self._valid_faucet.iteration,
                callbacks=[self._save_model]
            )
        except Exception as e:
            print("[*]Fail to training")
            print(e)

    def train_on_step


class SequentialTrainer:
    def __init__(self, trainers: dict) -> None:
        """
            trainers = [
                {"faucet", faucetU, "chain": [ModelA], "target": [ModelA], step=1},                  ごく普通のモデル学習
                {"faucet", faucetV, "chain": [ModelA, ModelB], "target": [ModelA, ModelB], step=2},  ただのmodel結合
                {"faucet", faucetW, "chain": [ModelB, ModelC], "target": [ModelB], step=3},          GANのGenerator学習
                {"faucet", faucetX, "chain": [ModelB, ModelC], "target": [ModelC], step=10},         GANのdiscriminator学習
            ]
        """
        self._trainers = trainers
        self._training_weights = training_weights

   def _validate(self):
        if len(self._trainers) != len(self._training_weights):
            raise ArgumentError
        if any([w <= 0 for w in self._training_weights]):
            raise ArgumentError

    def __call__(self, step: int) -> None:
        for i in range(step):
            for trainer, s in zip(self._trainers, self._training_weights):
                for i in range(s):
                    trainer.train_on_step()


def seaquential_train():
    def validate_args():
        if "NOT HAVE SAME keys xxx_pair":
            raise ValueError

    def combine_models(model_chains):
        model_chains = {
            "model_x": {"output_model": ["model_y", "model_z"], "input_model": "しばらくnull"},
            "model_y": {"output_model": ["model_a"], "input_model": "しばらくnull"},
            "model_z": {"output_model": None, "input_model": "しばらくnull"},
            "model_a": {"output_model": None, "input_model": "しばらくnull"}
        }
        for i_chain in model_chains:

        amodels =

    validate_args()

    def pair_train():

    return None
