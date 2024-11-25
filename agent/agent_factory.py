from abc import ABCMeta
from .base_agent import BaseAgent

class AgentFactory:
    _registered_agents = {}

    @classmethod
    def register(cls, agent_class, *aliases):
        cls._registered_agents[agent_class.__name__] = agent_class
        for alias in aliases:
            if not issubclass(agent_class, BaseAgent):
                raise ValueError(f"Agent {{ {agent_class} }} must extend BaseAgent")
            cls._registered_agents[alias] = agent_class

    @classmethod
    def create_agent(cls, classname, **config) -> BaseAgent:
        agent_class:BaseAgent = cls._registered_agents.get(classname)
        if agent_class is None:
            raise ValueError(f"Unknown agent class: {classname}")
        return agent_class(**config)

# Decorador para registrar automáticamente las clases de agent con múltiples alias
def register_agent(*aliases):
    if len(aliases) == 1 and isinstance(aliases[0], ABCMeta):
        cls = aliases[0]
        AgentFactory.register(cls)
        return cls
    
    def decorator(cls):
        AgentFactory.register(cls, *aliases)
        return cls
    return decorator