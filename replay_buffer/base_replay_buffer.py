from abc import ABC, abstractmethod

class BaseReplayBuffer(ABC):
    """
    Clase base para todos los tipos de replay buffers.

    Esta clase define los métodos básicos que deben implementar todos los replay buffers,
    permitiendo una interfaz común para los agentes.
    """

    def __init__(self, capacity):
        """
        Inicializa el replay buffer.

        Args:
            capacity (int): Capacidad máxima del buffer.
            data_spec: Especificación de los datos que se almacenarán en el buffer.
        """
        self._capacity = capacity

    @property
    def capacity(self):
        """Devuelve la capacidad máxima del buffer."""
        return self._capacity

    @abstractmethod
    def add(self, state, action, reward, next_state, terminated, truncated):
        """
        Agrega un elemento al buffer.

        Args:
            element: Elemento a agregar.
        """
        pass

    @abstractmethod
    def sample(self, batch_size):
        """
        Muestra un batch aleatorio de elementos del buffer.

        Args:
            batch_size (int): Tamaño del batch.

        Returns:
            batch, indices, weights
        """
        pass

    @abstractmethod
    def __len__(self):
        """Devuelve el número de elementos actualmente en el buffer."""
        pass

    def update(self, indices, td_errors):
        """
        Método opcional para realizar actualizaciones en el buffer (por ejemplo, en buffers priorizados).
        """
        pass