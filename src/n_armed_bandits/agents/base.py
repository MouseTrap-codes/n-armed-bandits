from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, n: int):
        self.n = n

    @abstractmethod
    def select_action(self) -> int:
        pass

    @abstractmethod
    def update(self, action: int, reward: float):
        pass
