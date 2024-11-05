import jax.numpy as jnp
from jax import grad
from math import ceil
import numpy as np  # For NaN handling only
import matplotlib.pyplot as plt

class RasterData:
    """Handles raster data and associated properties."""
    def __init__(self, dataset_path, species_name, resolution):
        self.species_name = species_name
        self.data = self.load_data(dataset_path) / 100  # Adjust factor as needed
        self.resolution = resolution(self.data) / 1000  # Resolution in km

    @staticmethod
    def load_data(dataset_path):
        """Placeholder for data loading function."""
        # Implement actual raster data loading
        return jnp.zeros((100, 100))  # Example placeholder array


class WindowOperation:
    """Handles window-based operations on raster data."""
    def __init__(self, raster_data, window_size, D, cut_off):
        self.raster_data = raster_data
        self.window_size = window_size
        self.D = D
        self.cut_off = cut_off
        self.buffer_size = ceil(3 * D / self.raster_data.resolution)
        self.total_window_size = self.window_size + 2 * self.buffer_size
        self.output_array = jnp.full(self.raster_data.data.shape, jnp.nan)

    def replace_missing(self, array, value=jnp.nan):
        """Replace missing data in the array with specified value."""
        return jnp.where(jnp.isnan(array), value, array)

    def extract_window(self, x_start, y_start):
        """Extract a buffered window from the raster data."""
        window = self.raster_data.data[
            x_start:x_start + self.total_window_size,
            y_start:y_start + self.total_window_size
        ]
        return self.replace_missing(window)

    def iterate_windows(self):
        """Yield buffered windows for computation, skipping empty areas."""
        width, height = self.raster_data.data.shape
        x_steps = (width - 2 * self.buffer_size) // self.window_size
        y_steps = (height - 2 * self.buffer_size) // self.window_size

        for i in range(1, x_steps - 1):
            for j in range(1, y_steps - 1):
                x_start, y_start = i * self.window_size, j * self.window_size
                window = self.extract_window(x_start, y_start)
                
                if jnp.any(~jnp.isnan(window)) and jnp.any(window > self.cut_off):
                    yield x_start, y_start, window


class HabitatSensitivityModel:
    """Calculates sensitivity of habitat based on proximity and habitat quality."""
    def __init__(self, D, res):
        self.D = D
        self.res = res

    def calculate_euclidean_distance(self, graph):
        """Placeholder for calculating Euclidean distances on a grid graph."""
        return jnp.zeros(graph.shape)  # Replace with actual calculation

    def calculate_functional_habitat(self, habitat_quality, proximity_matrix):
        """Calculates functional habitat quality."""
        return jnp.dot(habitat_quality, proximity_matrix)  # Example placeholder calculation

    def compute_sensitivity(self, hab_qual, proximity_matrix):
        """Computes sensitivity based on habitat quality and proximity matrix."""
        q = jnp.array([hab_qual[idx] for idx in self.id_to_grid_coordinate_list()])
        
        # Calculate gradient of functional habitat loss with respect to habitat quality
        sensitivity_fn = grad(lambda q: self.calculate_functional_habitat(q, proximity_matrix))
        return sensitivity_fn(q)

    @staticmethod
    def id_to_grid_coordinate_list():
        """Generate grid coordinates for a graph."""
        # Placeholder for the actual grid coordinates retrieval logic
        return [(i, j) for i in range(10) for j in range(10)]


class SensitivityAnalysis:
    """Main class for running habitat sensitivity analysis over raster windows."""
    def __init__(self, raster_data, window_op, model):
        self.raster_data = raster_data
        self.window_op = window_op
        self.model = model

    def run_analysis(self):
        """Performs the sensitivity analysis on each valid window."""
        for x_start, y_start, hab_qual in self.window_op.iterate_windows():
            # Build grid graph and calculate Euclidean distances
            g = GridGraph(hab_qual, vertex_activities=hab_qual > self.window_op.cut_off)
            euclidean_distance = self.model.calculate_euclidean_distance(g)
            proximity_matrix = jnp.exp(-euclidean_distance / self.model.D)

            # Calculate sensitivities within the buffered window
            sensitivities_vec = self.model.compute_sensitivity(hab_qual, proximity_matrix)
            sensitivities = jnp.full(g.shape, jnp.nan)
            for v, idx in enumerate(self.model.id_to_grid_coordinate_list()):
                sensitivities = sensitivities.at[idx].set(sensitivities_vec[v])

            # Store results into the core window area of the output array
            core_range = slice(self.window_op.buffer_size, self.window_op.buffer_size + self.window_op.window_size)
            self.window_op.output_array = self.window_op.output_array.at[
                x_start + core_range, y_start + core_range
            ].set(sensitivities[core_range, core_range])

        return self.window_op.output_array

    def plot_results(self):
        """Plot the final output array."""
        plt.imshow(self.window_op.output_array, cmap="viridis")
        plt.colorbar(label="Sensitivity")
        plt.title("Habitat Sensitivity Map")
        plt.show()


# Example usage
if __name__ == "__main__":
    # Load dataset and set up parameters
    raster_data = RasterData(dataset_path="path/to/dataset", species_name="Salmo trutta", resolution=resolution)
    window_op = WindowOperation(raster_data=raster_data, window_size=40, D=1.0, cut_off=0.1)
    model = HabitatSensitivityModel(D=1.0, res=raster_data.resolution)
    
    # Run analysis
    analysis = SensitivityAnalysis(raster_data=raster_data, window_op=window_op, model=model)
    output_array = analysis.run_analysis()
    
    # Plot the results
    analysis.plot_results()
