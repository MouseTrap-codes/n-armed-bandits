from abc import ABC, abstractmethod
from typing import Optional


class Agent(ABC):
    def __init__(self, n: int):
        self.n = n

    @abstractmethod
    def select_action(self) -> Optional[int]:
        pass

    @abstractmethod
    def update(self, action: int, reward: float):
        pass
