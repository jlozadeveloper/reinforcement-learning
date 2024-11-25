from .base_replay_buffer import BaseReplayBuffer

class ReplayBufferFactory:
    _registered_buffers = {}

    @classmethod
    def register(cls, buffer_class, *aliases):
        cls._registered_buffers[buffer_class.__name__] = buffer_class
        for alias in aliases:
            if not issubclass(buffer_class, BaseReplayBuffer):
                raise ValueError(f"Buffer replay {{ {buffer_class} }} must extend BaseReplayBuffer")
            cls._registered_buffers[alias] = buffer_class

    @classmethod
    def create_buffer(cls, classname, **config) -> BaseReplayBuffer:
        buffer_class:BaseReplayBuffer = cls._registered_buffers.get(classname)
        if buffer_class is None:
            raise ValueError(f"Unknown replay buffer class: {classname}")
        return buffer_class(**config)

# Decorador para registrar automáticamente las clases de replay buffer con múltiples alias
def register_buffer(*aliases):
    def decorator(cls):
        ReplayBufferFactory.register(cls, *aliases)
        return cls
    return decorator