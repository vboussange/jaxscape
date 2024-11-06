from abc import ABC, abstractmethod


class Distance(ABC):
    @abstractmethod
    def distance(self):
        pass
