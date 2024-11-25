from abc import ABCMeta
from .base_trainer import BaseTrainer

class TrainerFactory:
    _registered_trainers = {}

    @classmethod
    def register(cls, trainer_class, *aliases):
        cls._registered_trainers[trainer_class.__name__] = trainer_class
        for alias in aliases:
            if not issubclass(trainer_class, BaseTrainer):
                raise ValueError(f"Trainer {{ {trainer_class} }} must extend BaseTrainer")
            cls._registered_trainers[alias] = trainer_class

    @classmethod
    def create_trainer(cls, classname, **config) -> BaseTrainer:
        trainer_class:BaseTrainer = cls._registered_trainers.get(classname)
        if trainer_class is None:
            raise ValueError(f"Unknown trainer class: {classname}")
        return trainer_class(**config)

# Decorador para registrar automáticamente las clases de trainer con múltiples alias
def register_trainer(*aliases):
    if len(aliases) == 1 and isinstance(aliases[0], ABCMeta):
        cls = aliases[0]
        TrainerFactory.register(cls)
        return cls
    
    def decorator(cls):
        TrainerFactory.register(cls, *aliases)
        return cls
    return decorator