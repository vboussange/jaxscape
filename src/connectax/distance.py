from abc import ABC, abstractmethod
import equinox as eqx


class AbstractDistance(eqx.Module):
    @abstractmethod
    def __call__(self):
        pass
