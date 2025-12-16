from abc import ABC, abstractmethod
from typing import Optional
import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .gridgraph import AbstractGraph

class AbstractDistance(eqx.Module):
    """
    Abstract base class for distance computations on graphs.
    
    This class defines the interface for computing various types of distances on any graph type. Subclasses must implement the abstract methods to provide specific distance calculations.
    
    The __call__ method provides a unified interface with the following usage patterns:
    
    - Use `nodes` for pairwise distances among a specified set of nodes.
    
    - Use `sources` and `targets` for distances from sources to targets, which is efficient for scenarios like computing distances from all nodes to a single target (e.g., using specialized algorithms such as Bellman-Ford).
    
    Parameters:
        graph: The graph on which to compute distances.
        sources: Optional array of source node indices. If provided with targets, computes distances from sources to targets.
        targets: Optional array of target node indices. If provided with sources, computes distances from sources to targets.
        nodes: Optional array of node indices. If provided, computes pairwise distances among these nodes.
    
    Returns:
        Distance matrix or array depending on the method called.
        
    !!! example

        ```python
        # All-pairs distance
        dist_matrix = distance(graph)  # Shape: (n_nodes, n_nodes)

        # Source-target distance
        dist_matrix = distance(graph, sources=[0, 1], targets=[10, 20])  # Shape: (2, 2)

        # Pairwise among subset
        dist_matrix = distance(graph, nodes=[0, 5, 10])  # Shape: (3, 3)
        ```
    """
    def __call__(
        self, 
        graph: AbstractGraph, 
        sources: Optional[Array] = None, 
        targets: Optional[Array] = None, 
        nodes: Optional[Array] = None
    ) -> Array:
        if nodes is not None:
            assert sources is None and targets is None, "Specify either `nodes`, `sources`, `targets`, or both `sources` and `targets`."
            nodes = graph.coord_to_index(nodes[:, 0], nodes[:, 1]) if nodes.ndim == 2 else nodes
            return self.nodes_to_nodes_distance(graph, nodes)
        
        elif sources is not None or targets is not None:
            sources = jnp.arange(graph.nv) if sources is None else sources
            targets = jnp.arange(graph.nv) if targets is None else targets

            sources = graph.coord_to_index(sources[:, 0], sources[:, 1]) if sources.ndim == 2 else sources
            targets = graph.coord_to_index(targets[:, 0], targets[:, 1]) if targets.ndim == 2 else targets

            return self.sources_to_targets_distance(graph, sources, targets)
        
        else:
            return self.all_pairs_distance(graph)

    @abstractmethod
    def nodes_to_nodes_distance(self, graph: AbstractGraph, nodes: Array) -> Array:
        pass

    @abstractmethod
    def sources_to_targets_distance(self, graph: AbstractGraph, sources: Array, targets: Array) -> Array:
        pass
        
    @abstractmethod
    def all_pairs_distance(self, graph: AbstractGraph) -> Array:
        pass
        
        
