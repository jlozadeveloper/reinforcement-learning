from .base_policy import BasePolicy

class PolicyFactory:
    _registered_policies = {}

    @classmethod
    def register(cls, policy_class, *aliases):
        cls._registered_policies[policy_class.__name__] = policy_class
        for alias in aliases:
            if not issubclass(policy_class, BasePolicy):
                raise ValueError(f"Policy {{ {policy_class} }} must extend BasePolicy")
            cls._registered_policies[alias] = policy_class

    @classmethod
    def create_policy(cls, classname, **config) -> BasePolicy:
        policy_class:BasePolicy = cls._registered_policies.get(classname)
        if policy_class is None:
            raise ValueError(f"Unknown policy class: {classname}")
        return policy_class(**config)

# Decorador para registrar automáticamente las clases de replay buffer con múltiples alias
def register_policy(*aliases):
    def decorator(cls):
        PolicyFactory.register(cls, *aliases)
        return cls
    return decorator