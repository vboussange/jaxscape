from abc import ABC, abstractmethod


class Distance(ABC):
    @abstractmethod
    def get_distance_matrix(self):
        pass
