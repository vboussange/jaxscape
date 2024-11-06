from connectax.gridgraph import GridGraph, ROOK_CONTIGUITY
import jax.numpy as jnp

class Landscape(GridGraph):
    def __init__(self, habitat_quality, proximity, activities=None):
        """
        A landscape, which takes in a proximity matrix `K` and a habitat quality raster `habitat_quality`.
        """
        if activities == None:
            activities = habitat_quality > 0
        assert activities.shape == habitat_quality.shape
        assert activities.sum() == proximity.shape[0]
        super().__init__(activities, habitat_quality)
        self.proximity = proximity
        
    @property
    def get_adjacency_matrix(self):
        return self.proximity
    
    def functional_habitat(self):
        active_ij = self.active_vertex_index_to_coord(jnp.arange(self.nb_active()))
        q = self.vertex_weights[active_ij[:,0], active_ij[:,1]]
        K = self.proximity
        return q @ (K @ q) 