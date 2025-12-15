from abc import ABC, abstractmethod
import equinox as eqx
import jax.numpy as jnp

class AbstractDistance(eqx.Module):
    """
    Abstract base class for distance computations on graphs.
    
    This class defines the interface for computing various types of distances on a grid graph. Subclasses must implement the abstract methods to provide specific distance calculations.
    
    The __call__ method provides a unified interface with the following usage patterns:
    
    - Use `nodes` for pairwise distances among a specified set of nodes.
    
    - Use `sources` and `targets` for distances from sources to targets, which is efficient for scenarios like computing distances from all nodes to a single target (e.g., using specialized algorithms such as Bellman-Ford).
    
    Parameters:
        grid: The grid graph on which to compute distances.
        sources: Optional array of source node indices. If provided with targets, computes distances from sources to targets.
        targets: Optional array of target node indices. If provided with sources, computes distances from sources to targets.
        nodes: Optional array of node indices. If provided, computes pairwise distances among these nodes.
    
    Returns:
        Distance matrix or array depending on the method called.
        
    !!! example

        ```python
        # All-pairs distance
        dist_matrix = distance(grid)  # Shape: (n_nodes, n_nodes)

        # Source-target distance
        dist_matrix = distance(grid, sources=[0, 1], targets=[10, 20])  # Shape: (2, 2)

        # Pairwise among subset
        dist_matrix = distance(grid, nodes=[0, 5, 10])  # Shape: (3, 3)
        ```
    """
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
        
        
