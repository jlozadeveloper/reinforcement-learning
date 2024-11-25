import numpy as np
from abc import ABC, abstractmethod
from keras import Model

class BaseModel(ABC):
    @classmethod
    @abstractmethod
    def create_model(cls, input_shape, output_shape, **config) -> Model:
        raise NotImplementedError