import numpy as np
from .base_replay_buffer import BaseReplayBuffer
from .replay_buffer_factory import register_buffer


@register_buffer('prioritized', 'prioritized-buffer')
class PrioritizedReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self._buffer = []
        self._position = 0
        self._priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, reward, next_state, terminated, truncated):
        max_priority = self._priorities.max() if self._buffer else 1.0
        element = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "terminated": terminated,
            "truncated": truncated
        }
        if len(self._buffer) < self._capacity:
            self._buffer.append(element)
        else:
            self._buffer[self._position] = element
        self._priorities[self._position] = max_priority
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size):
        priorities = self._priorities[:len(self._buffer)] ** self.alpha
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self._buffer), batch_size, p=probabilities)
        batch = [self._buffer[i] for i in indices]

        weights = (len(self._buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        return batch, indices, weights

    def update(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self._priorities[idx] = abs(td_error) + 1e-6

    def __len__(self):
        return len(self._buffer)
    
# class PrioritizedReplayBuffer:
#     def __init__(self, capacity, alpha=0.6):
#         self.capacity = capacity
#         self.alpha = alpha
#         self.buffer = []
#         self.priorities = []
#         self.position = 0

#     def add(self, state, action, reward, next_state, done):
#         max_priority = max(self.priorities, default=1.0)
#         if len(self.buffer) < self.capacity:
#             self.buffer.append((state, action, reward, next_state, done))
#             self.priorities.append(max_priority)
#         else:
#             self.buffer[self.position] = (state, action, reward, next_state, done)
#             self.priorities[self.position] = max_priority
#         self.position = (self.position + 1) % self.capacity

#     def sample(self, batch_size):
#         priorities = np.array(self.priorities) ** self.alpha
#         probabilities = priorities / priorities.sum()
#         indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
#         batch = [self.buffer[idx] for idx in indices]
        
#         weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
#         weights /= weights.max()  # Normalizamos los pesos para estabilidad
#         return batch, indices, weights
