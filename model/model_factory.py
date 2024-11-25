from abc import ABCMeta
from .base_model import BaseModel

class ModelFactory:
    _registered_models = {}

    @classmethod
    def register(cls, model_class, *aliases):
        cls._registered_models[model_class.__name__] = model_class
        for alias in aliases:
            if not issubclass(model_class, BaseModel):
                raise ValueError(f"Model {{ {model_class} }} must extend BaseModel")
            cls._registered_models[alias] = model_class

    @classmethod
    def create_model(cls, classname, **config) -> BaseModel:
        model_class:BaseModel = cls._registered_models.get(classname)
        if model_class is None:
            raise ValueError(f"Unknown model classname: {classname}")
        return model_class.create_model(**config)

# Decorador para registrar automáticamente las clases de model con múltiples alias
def register_model(*aliases):
    if len(aliases) == 1 and isinstance(aliases[0], ABCMeta):
        cls = aliases[0]
        ModelFactory.register(cls)
        return cls
    
    def decorator(cls):
        ModelFactory.register(cls, *aliases)
        return cls
    return decorator