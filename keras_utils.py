"""Module containing utility classes and routines used in training of policies"""
import functools
from typing import List, Any, Tuple

import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from tensorflow.config import list_physical_devices
from tensorflow.keras.models import (Model,load_model)
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import (
    EarlyStopping,
    CSVLogger,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.metrics import top_k_categorical_accuracy
from scipy import sparse

top10_acc = functools.partial(top_k_categorical_accuracy, k=10)
top10_acc.__name__ = "top10_acc"  # type: ignore

top50_acc = functools.partial(top_k_categorical_accuracy, k=50)
top50_acc.__name__ = "top50_acc"  # type: ignore


class InMemorySequence(Sequence):  # pylint: disable=W0223
    """
    Class for in-memory data management

    :param input_filname: the path to the model input data
    :param output_filename: the path to the model output data
    :param batch_size: the size of the batches
    """

    def __init__(
        self, input_filename: str, output_filename: str, batch_size: int
    ) -> None:
        self.batch_size = batch_size
        self.input_matrix = self._load_data(input_filename)
        self.label_matrix = self._load_data(output_filename)
        self.input_dim = self.input_matrix.shape[1]

    def __len__(self) -> int:
        return int(np.ceil(self.label_matrix.shape[0] / float(self.batch_size)))

    def _make_slice(self, idx: int) -> slice:
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")

        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        return slice(start, end)

    @staticmethod
    def _load_data(filename: str) -> np.ndarray:
        try:
            return sparse.load_npz(filename)
        except ValueError:
            return np.load(filename)["arr_0"]
class PrintLearningRateCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 400 == 0 :
            # Get the current learning rate from the optimizer
            lr = self.model.optimizer.learning_rate
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                # If the learning rate is a schedule, get the current value
                lr = lr(self.model.optimizer.iterations)
            elif isinstance(lr, tf.Variable):
                # If the learning rate is a variable, get its value
                lr = lr.numpy()
            print(f'lr {lr}')

def setup_callbacks(
    log_filename: str, checkpoint_filename: str
) -> Tuple[CSVLogger,  ModelCheckpoint]:
    """
    Tuple[EarlyStopping, CSVLogger,  ModelCheckpoint, ReduceLROnPlateau]
    Setup Keras callback functions: early stopping, CSV logger, model checkpointing,
    and reduce LR on plateau

    :param log_filename: the filename of the CSV log
    :param checkpoint_filename: the filename of the checkpoint
    :return: all the callbacks
    """

    csv_logger = CSVLogger(log_filename)
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_filename,
        save_weight_only = False,
        save_format = "tf",
        monitor="val_loss",
        save_best_only=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=0,
        mode="auto",
        min_delta=0.000001,
        cooldown=10,
        min_lr=0,
    )

    print_lr = PrintLearningRateCallback()
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)




    return [csv_logger, checkpoint]

class WarmUpLearningRateScheduler(LearningRateSchedule):

    def __init__(self,init_lr,target_lr,warmup_steps,warmup_target,alpha,decay_steps,total_step_number):
        super().__init__()
        self.init_lr = init_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        self.decay_steps = decay_steps
        self.warmup_target = warmup_target
        self.total_step_number = total_step_number
    @tf.function
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.init_lr + (self.warmup_target - self.init_lr) * (step / self.warmup_steps)
        if step < self.warmup_steps:
            return warmup_lr
        else:

            global_decay = (-1/self.total_step_number)*step + 1

            frac = (step / (self.decay_steps*2+200*tf.math.sqrt(step))) % 0.5
            cosine_decay = tf.cos(tf.constant(np.pi) * frac)
            cyclical_decay = (1 - self.alpha) * cosine_decay + self.alpha

            lr = self.target_lr * cyclical_decay * global_decay
            return lr

    def get_config(self):
        return {
            'init_lr': self.init_lr,
            'target_lr': self.target_lr,
            'warmup_steps': self.warmup_steps,
            'warmup_target' : self.warmup_target,
            'alpha': self.alpha,
            'decay_steps': self.decay_steps
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def train_keras_model(
    model: Model,
    train_seq: InMemorySequence,
    valid_seq: InMemorySequence,
    loss: str,
    metrics: List[Any],
    callbacks: List[Any],
    epochs: int,
) -> None:
    """
    Train a Keras model, but first compiling it and then fitting
    it to the given data

    :param model: the initialized model
    :param train_seq: the training data
    :param valid_seq: the validation data
    :param loss: the loss function
    :param metrics: the metric functions
    :param callbacks: the callback functions
    :param epochs: the number of epochs to use
    """
    print(f"Available GPUs: {list_physical_devices('GPU')}")
    lr_scheduler = WarmUpLearningRateScheduler(init_lr=8e-8,warmup_target=6e-4,target_lr=6e-4,warmup_steps=20000,alpha=0,decay_steps=20000,total_step_number=6e5)
    weight_decay_scheduler = WarmUpLearningRateScheduler(init_lr=8e-9,warmup_target=6e-5,target_lr=6e-5,warmup_steps=20000,alpha=0,decay_steps=20000,total_step_number=6e5)
    adam = tfa.optimizers.AdamW(learning_rate=lr_scheduler, beta_1=0.9, beta_2=0.999,weight_decay=weight_decay_scheduler)
    #original optimizer :
    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    model.compile(
        optimizer=adam,
        loss=loss,
        metrics=metrics,
    )

    model.fit(
        train_seq,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=valid_seq,
        max_queue_size=20,
        workers=1,
        use_multiprocessing=False,
    )
    """
    loaded_model = load_model('model')
    loaded_model.save('keras_model.hdf5')"""
