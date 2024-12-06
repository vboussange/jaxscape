from abc import ABC, abstractmethod
import equinox as eqx

# TODO: in the future, __call__ should dispatch to specialized methods depending on 
# whether the user provides sources and or targets. It should also handle the cases where 
# the user provides vertex index (int) or coordinates (tuple of int), similar to LCPDistance
class AbstractDistance(eqx.Module):
    @abstractmethod
    def __call__(self, grid, sources, targets):
        pass
