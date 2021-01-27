import os
from abc import ABC, abstractmethod, ABCMeta
import torch
from torch import nn

from  ._types_ import *


"""
Abstract Model class
"""
class AbstractModel(nn.Module):
    __metaclass__ = ABCMeta
    def __init__(self, filepath=None, **kwargs) -> None:
        super(AbstractModel, self).__init__()
        self.filepath = filepath

    def __repr__(self):
        pass

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def save(self):
        """
        Saves the model
        :return: None
        """
        pass

    @abstractmethod
    def save_checkpoint(self, epoch_num):
        """
        Saves the model checkpoints
        :param epoch_num: int,
        :return: None
        """
        pass

    @abstractmethod
    def load(self, cpu=False):
        """
        Loads the model
        :param cpu: bool, specifies if the model should be loaded on the CPU
        :return: None
        """
        pass

    @abstractmethod
    def initialization(self):
        """
        Initializes the network params
        default xavier initialization and kaiming initialization
        :return:
        """
        pass

    @abstractmethod
    def objective_func(self, *inputs: Any, **kwargs) -> Tensor:
        """
        objective function for updating weights for neurons
        :return:
        """
        raise NotImplementedError

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

class AbtractTrainer(ABC):
    """
    Abstract base class which will serve as a NN trainer
    """
    def __init__(self, dataset,
                 model,
                 lr=1e-4):
        """
        Initializes the trainer class
        :param dataset: torch Dataset object
        :param model: torch.nn object
        :param lr: float, learning rate
        """
        self.dataset = dataset
        self.model = model
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        self.global_iter = 0
        self.trainer_config = ''
        self.writer = None