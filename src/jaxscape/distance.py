from abc import ABC, abstractmethod
import equinox as eqx
import jax.numpy as jnp

class AbstractDistance(eqx.Module):
    def __call__(self, grid, sources=None, targets=None, nodes=None):
        if nodes is not None:
            assert sources is None and targets is None, "Specify either `nodes`, `sources`, `targets`, or both `sources` and `targets`."
            nodes = grid.coord_to_index(nodes[:, 0], nodes[:, 1]) if nodes.ndim == 2 else nodes
            return self.nodes_to_nodes_distance(grid, nodes)
        
        elif sources is not None or targets is not None:
            sources = jnp.arange(grid.nv) if sources is None else sources
            targets = jnp.arange(grid.nv) if targets is None else targets

            sources = grid.coord_to_index(sources[:, 0], sources[:, 1]) if sources.ndim == 2 else sources
            targets = grid.coord_to_index(targets[:, 0], targets[:, 1]) if targets.ndim == 2 else targets

            return self.sources_to_targets_distance(grid, sources, targets)
        
        else:
            return self.all_pairs_distance(grid)

    @abstractmethod
    def nodes_to_nodes_distance(self, grid, nodes):
        pass

    @abstractmethod
    def sources_to_targets_distance(self, grid, sources, targets):
        pass
        
    @abstractmethod
    def all_pairs_distance(self, grid):
        pass
        
        
