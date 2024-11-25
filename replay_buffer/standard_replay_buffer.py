import numpy as np
from .base_replay_buffer import BaseReplayBuffer
from .replay_buffer_factory import register_buffer

@register_buffer('standard', 'standard-buffer', 'standardbuffer')
class StandardReplayBuffer(BaseReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffer = []
        self._position = 0

    def add(self, state, action, reward, next_state, terminated, truncated):
        """Agrega un elemento al buffer, sobrescribiendo el más antiguo si es necesario."""
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
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size) -> list[dict]:
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]
        return batch, indices, None

    def __len__(self):
        return len(self._buffer)