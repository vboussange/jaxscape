"""Inverse landscape modelling with ESA WorldCover.

This script fits a simple categorical resistance model to genetic
differentiation (Fst-like) data for the CESAR sites, using JAXScape
resistance distances and Optimistix for gradient-based optimization.

High-level steps
----------------
1. Load and cache ESA WorldCover land-cover around the CESAR sites.
2. Downsample and convert land-cover classes to compact integer ids.
3. Build a GridGraph and associate each CESAR site with a grid node.
4. Define a one-layer categorical model mapping land-cover to
  positive resistance.
5. Train the model to align resistance distances with observed
  genetic distances.
6. Evaluate the fit by comparing predicted vs target distances.

NOTE: when loss is above 0.0018, the network fails to learn meaningful representations.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from equinox import nn
import equinox as eqx
import optimistix as optx
from jaxscape import GridGraph, ResistanceDistance, LCPDistance
from jaxscape.solvers import CholmodSolver
from preprocess import load_cesar_data
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

LANDCOVER_PATH = 'landcover_7855.tif'
SITE_METADATA_PATH = 'cesar_site_metadata.gpkg'
GENETIC_DISTANCES_PATH = 'cesar_genetic_distances.npy'

# Coarsening factor applied to the *continuous* embeddings/one-hot features.
COARSENING_FACTOR = 10
SOLVER = CholmodSolver()
DISTANCE_FUN = ResistanceDistance(solver=SOLVER)
MAX_STEPS = 500
# Alternative distance for experiments:
# DISTANCE_FUN = LCPDistance()

def load_data():
    predictor_raster = rioxarray.open_rasterio(
        CACHED_PREDICTORS, mask_and_scale=True, band_as_variable=True
    )
    site_metadata = gpd.read_file(SITE_METADATA_PATH)
    target_distances = np.load(GENETIC_DISTANCES_PATH)
    site_gdf = site_metadata.to_crs(epsg=7855)
    return predictor_raster, target_distances, site_gdf



def prepare_feature_targets(predictor_raster, site_gdf):
  """Create model inputs and graph structures from land-cover and sites.

  Returns
  -------
  features : jax.numpy.ndarray
      Integer class ids for each grid cell.
  unique_classes : jax.numpy.ndarray
      Mapping back to original WorldCover class values.
  target_nodes : jax.numpy.ndarray
      Node indices corresponding to CESAR sampling locations.
  grid : GridGraph
      Graph built on a constant-resistance grid, used only for indexing.
  """
  # Downstream model will operate on *continuous* (one-hot) features.
  # First compress sparse WorldCover classes to contiguous ids 0..K-1.
  raw_band = np.asarray(predictor_raster["band_1"])
  unique_vals, inverse = np.unique(raw_band.ravel(), return_inverse=True)
  class_ids = inverse.reshape(raw_band.shape).astype(np.int32)
  features_categorical = jnp.array(class_ids).squeeze()
  unique_classes = jnp.array(unique_vals)

  # One-hot encode and *then* coarsen using mean pooling so that a
  # coarse cell represents the average class-composition of underlying
  # fine cells.
  features_onehot = jax.nn.one_hot(
    features_categorical, num_classes=len(unique_vals)
  )  # (H, W, K)
  # Reorder axes to (K, H, W) for xarray-like coarsening
  features_onehot = jnp.moveaxis(features_onehot, -1, 0)
  # Wrap in DataArray-like structure via rioxarray/xarray for coarsen
  # using the landcover grid for coordinates.
  coarsen_x = COARSENING_FACTOR
  coarsen_y = COARSENING_FACTOR
  # Broadcast coordinates from predictor_raster to feature grid
  coords = {
    "band": np.arange(features_onehot.shape[0]),
    "y": predictor_raster.y.values,
    "x": predictor_raster.x.values,
  }
  feature_da = rioxarray.open_rasterio(
    predictor_raster.rio.to_raster,  # dummy to satisfy interface; we only need coords
  ) if False else None  # placeholder to avoid misuse; we will build xarray manually

  import xarray as xr

  feature_da = xr.DataArray(
    features_onehot,
    coords=coords,
    dims=("band", "y", "x"),
  )
  feature_da = feature_da.coarsen(x=coarsen_x, y=coarsen_y, boundary="trim").mean()
  # Move back to (H, W, K)
  features_onehot_coarse = jnp.moveaxis(feature_da.data, 0, -1)

  site_gdf = site_gdf.to_crs(epsg=7855)

  # Map site coordinates to nearest *coarsened* raster indices
  x_idx = jnp.array(
    [int(np.argmin(np.abs(feature_da.x.values - x))) for x in site_gdf.geometry.x.values]
  )
  y_idx = jnp.array(
    [int(np.argmin(np.abs(feature_da.y.values - y))) for y in site_gdf.geometry.y.values]
  )

  grid = GridGraph(jnp.ones((feature_da.x.size, feature_da.y.size)), fun=lambda x, y: (x + y) / 2)
  target_nodes = grid.coord_to_index(x_idx, y_idx)
  return features_onehot_coarse, unique_classes, target_nodes, grid, feature_da


def build_model(num_classes: int, seed: int = 1) -> tuple[eqx.Module, tuple, tuple]:
  key = jax.random.PRNGKey(seed)

  class ResistanceModel(eqx.Module):
    layers: list
    num_classes: int

    def __init__(self, num_classes: int, key):
      self.num_classes = num_classes
      k1, k2, k3 = jax.random.split(key, 3)
      hidden_dim = 16  # can tune, but increasing to 32 degrades performance

      self.layers = [
        nn.Linear(num_classes, hidden_dim, key=k1),
        jax.nn.relu,
        nn.Linear(hidden_dim, hidden_dim, key=k2),
        jax.nn.relu,
        nn.Linear(hidden_dim, 1, key=k3),
      ]

    def __call__(self, x):
      # x: one-hot feature vector
      for layer in self.layers:
        x = layer(x)
      return jnp.exp(x) + 1e-3 # ensure positive resistance

  model = ResistanceModel(num_classes, key)
  params, static = eqx.partition(model, eqx.is_inexact_array)
  return model, params, static


def visualise_initial_prediction(resistance_field):
  fig, ax = plt.subplots()
  im = ax.imshow(resistance_field, cmap="RdYlGn")
  ax.set_title("Initial model resistance prediction")
  ax.axis("off")
  plt.colorbar(im, ax=ax, label="resistance")
  plt.show()


# -----------------------------------------------------------------------------

# Data loading and preprocessing
# -----------------------------------------------------------------------------
predictor_raster, genetic_distances, site_gdf = load_reproject_crop_raw_feature_labels()

fig, ax = plt.subplots()
predictor_raster["band_1"].plot(ax=ax, cmap="tab20", add_colorbar=True)
ax.scatter(site_gdf.geometry.x, site_gdf.geometry.y, c="red", s=50, label="Sites")
plt.show()

features_onehot, unique_classes, target_nodes, ref_grid, coarse_feature_da = prepare_feature_targets(
  predictor_raster, site_gdf
)

node_coords = ref_grid.index_to_coord(target_nodes)
x_indices, y_indices = node_coords[:, 0], node_coords[:, 1]

fig, ax = plt.subplots()
ax.imshow(features_onehot.argmax(axis=-1), cmap="tab20")
ax.scatter(x_indices, y_indices, c="blue", s=50, label="Sites")
for xi, yi, name in zip(x_indices, y_indices, site_gdf["site_name"].values):
  ax.text(
    int(xi),
    int(yi) - 5,
    str(name),
    color="black",
    fontsize=8,
    ha="center",
    va="bottom",
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
  )
plt.show()


# -----------------------------------------------------------------------------
# Model definition and training
# -----------------------------------------------------------------------------
model, params, static = build_model(len(unique_classes))
model_vmapped = jax.vmap(jax.vmap(model, in_axes=0), in_axes=0)

initial_resistance = model_vmapped(features_onehot).squeeze()
visualise_initial_prediction(initial_resistance)


@eqx.filter_jit
def loss_fn(params, args):
  """Squared error between target and predicted pairwise distances.

  The signature is constrained by Equinox/Optimistix; we therefore pass
  the full static context via ``args``.
  """
  static, features, target_flat_train, tri_i_train, tri_j_train = args
  model = eqx.combine(params, static)
  model_vmapped = jax.vmap(jax.vmap(model, in_axes=0), in_axes=0)
  resistance = model_vmapped(features).squeeze()
  grid = GridGraph(resistance, fun=lambda x, y: (x + y) / 2)
  predicted_distances = DISTANCE_FUN(grid, nodes=target_nodes)
  pred_flat_train = predicted_distances[tri_i_train, tri_j_train]
  return ((target_flat_train - pred_flat_train) ** 2).mean()


# Precompute all pair indices once (upper triangle) and build flat targets
n_sites = genetic_distances.shape[0]
tri_i_all, tri_j_all = np.triu_indices(n_sites, k=1)
target_flat_all = np.asarray(genetic_distances)[tri_i_all, tri_j_all]

# Train/test split on flattened pairs
(target_flat_train,
 target_flat_test,
 tri_i_train,
 tri_i_test,
 tri_j_train,
 tri_j_test) = train_test_split(
  target_flat_all,
  tri_i_all,
  tri_j_all,
  test_size=0.2,
  random_state=42,
)

tri_i_train = jnp.array(tri_i_train)
tri_j_train = jnp.array(tri_j_train)
tri_i_test = np.array(tri_i_test)
tri_j_test = np.array(tri_j_test)

# Sanity-check forward + loss on training pairs
_ = loss_fn(params, (static, features_onehot, target_flat_train, tri_i_train, tri_j_train))

solver = optx.LBFGS(rtol=1e-5, atol=1e-5, verbose=frozenset({"loss"}))
start_train_time = time.time()
opt_solution = optx.minimise(
  loss_fn,
  solver,
  params,
  args=(static, features_onehot, target_flat_train, tri_i_train, tri_j_train),
  max_steps=MAX_STEPS,
)
print(f"Training time: {time.time() - start_train_time:.2f} seconds")
print(opt_solution.stats)


fitted_model = eqx.combine(opt_solution.value, static)
fitted_vmapped = jax.vmap(jax.vmap(fitted_model, in_axes=0), in_axes=0)
fitted_resistance = fitted_vmapped(features_onehot).squeeze()

fig, ax = plt.subplots(figsize=(4, 5))
im = ax.imshow(fitted_resistance, cmap="RdYlGn")
ax.set_title("Fitted resistance")
ax.axis("off")
plt.colorbar(im, ax=ax, shrink=0.5, label="resistance")
plt.show()


# -----------------------------------------------------------------------------
# Evaluation: predicted vs target distances
# -----------------------------------------------------------------------------
pred_grid = GridGraph(fitted_resistance, fun=lambda x, y: (x + y) / 2)
pred_distances = DISTANCE_FUN(pred_grid, nodes=target_nodes)

genetic_np = np.asarray(genetic_distances)
pred_np = np.asarray(pred_distances)

# Flatten predictions for train and test pairs separately
train_pred = pred_np[tri_i_train, tri_j_train]
test_pred = pred_np[tri_i_test, tri_j_test]
train_target = target_flat_train
test_target = target_flat_test

# Metrics: R^2 and RMSE
r2_train = r2_score(train_target, train_pred)
r2_test = r2_score(test_target, test_pred)
rmse_train = root_mean_squared_error(train_target, train_pred)
rmse_test = root_mean_squared_error(test_target, test_pred)

print("Resistance-distance model fit (train/test)")
print(f"Train R^2 = {r2_train:.3f}, RMSE = {rmse_train:.4f}")
print(f"Test  R^2 = {r2_test:.3f}, RMSE = {rmse_test:.4f}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(train_pred, train_target, s=15, alpha=0.5, edgecolor="none", label="Train")
ax.scatter(test_pred, test_target, s=30, alpha=0.8, edgecolor="none", label="Test")

min_val = min(pred_np.min(), genetic_np.min())
max_val = max(pred_np.max(), genetic_np.max())
ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, label="1:1 line")

ax.set_xlabel("Predicted resistance distance")
ax.set_ylabel("Target genetic distance (Fst)")
ax.set_title("Resistance model: Fst vs effective resistance distance")

textstr = (
  f"Train R^2 = {r2_train:.3f}, RMSE = {rmse_train:.4f}\n"
  f"Test  R^2 = {r2_test:.3f}, RMSE = {rmse_test:.4f}"
)
ax.text(
  0.05,
  0.95,
  textstr,
  transform=ax.transAxes,
  fontsize=9,
  verticalalignment="top",
  bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
)

ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


