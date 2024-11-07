from abc import ABC, abstractmethod


class Distance(ABC):
    @abstractmethod
    def __call__(self):
        pass
