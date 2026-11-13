"""Generate the JAXScape logo.

The title is a barrier on a raster landscape of habitat. Its letters are
bitmap glyphs on a grid of square cells; their pixels have no habitat quality
and a low permeability, and every other pixel is habitat of quality and
permeability 1. A thin band of habitat runs around the title, so its strokes
split the habitat into compartments that the band barely connects.

Each iteration runs `SensitivityAnalysis` with resistance distance and ranks
corridors through the letters. A corridor is a run of one or two barrier cells
along a row or a column of the grid, with habitat on both sides: a straight cut
across a stroke. Restoring it raises its permeability to that of habitat, and
its first-order gain of connectivity is the sum of the elasticities,
`permeability * sensitivity`, of its pixels. Each iteration restores the
corridors of highest gain that lie at least `SPACING` pixels from the restored
ones, then runs the analysis again.

The frames show the raster in the colours of a land cover map: habitat as
forest, shaded by the elasticity of its pixels in flat bands; the title as
developed land; restored corridors as new vegetation.

Run `python make_logo.py` in this directory. It writes the animation
`logo.gif` and its last frame `logo.png`, in under a minute on one GPU. The
letters are defined below, `SpielmanApproximation` draws its random
projections from a fixed seed and GPU reductions are deterministic: the script
reads no input file, and every run writes the same files on a given machine.
"""

import os
import time


# Deterministic GPU reductions: every run on a machine draws the same frames.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
)

import jax
import jax.numpy as jnp
import numpy as np
from jaxscape import ResistanceDistance, SensitivityAnalysis, SpielmanApproximation
from jaxscape.solvers import GridMultigridSolver
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates


# Land cover colours: forest from low to high elasticity, developed land,
# new vegetation
HABITAT = ["#143625", "#1e4d31", "#2c673c", "#4a8547", "#7aa85a"]
BARRIER = "#dcd6c8"
CORRIDOR = "#a3e635"

# Bitmap glyphs: capitals take rows 0 to 6, lower case rows 2 to 6 and
# descenders rows 7 and 8.
GLYPHS = {
    "J": ("....#", "....#", "....#", "....#", "#...#", "#...#", ".###."),
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "X": ("#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"),
    "s": (".####", "#....", ".###.", "....#", "####."),
    "c": (".####", "#....", "#....", "#....", ".####"),
    "a": (".###.", "....#", ".####", "#...#", ".####"),
    "p": ("####.", "#...#", "#...#", "#...#", "####.", "#....", "#...."),
    "e": (".###.", "#...#", "#####", "#....", ".####"),
}
WORD = "JAXscape"
ROWS = 9

# Landscape
CELL = 14  # pixels along a side of a glyph cell
MARGIN = (10, 16)  # pixels of habitat above and below, and beside, the title
BARRIER_PERMEABILITY = 0.03

# Analysis
DISTANCE = ResistanceDistance(
    solver=GridMultigridSolver(rtol=1e-3),
    method=SpielmanApproximation(epsilon=0.2, seed=0),
)
DEPENDENCY_RANGE = 64  # pixels
COARSENING = 0.1  # blocks of 7 x 7 pixels

# Restoration
ITERATIONS = 8
CORRIDORS_PER_ITERATION = 3
MAX_CORRIDOR_CELLS = 2
SPACING = CELL  # pixels between corridors

# Rendering
SCALE = 2  # screen pixels per raster pixel
SMOOTHING = 6  # pixels: standard deviation of the smoothing of elasticity
FADE_FRAMES = 3  # frames from one iteration to the next
FADE_MS, HOLD_MS = 40, 260  # each fading frame; each iteration
FIRST_MS, LAST_MS = 700, 1800


def glyph_cells(word: str, gap: int = 1) -> np.ndarray:
    """Cells of the title, with `gap` columns of cells between letters."""
    columns = []
    for letter in word:
        glyph = GLYPHS[letter]
        top = 0 if letter.isupper() else 2
        block = np.zeros((ROWS, len(glyph[0])), bool)
        for r, row in enumerate(glyph):
            block[top + r] = [c == "#" for c in row]
        columns += [block, np.zeros((ROWS, gap), bool)]
    return np.concatenate(columns[:-1], axis=1)


def landscape(cells: np.ndarray) -> np.ndarray:
    """Barrier mask of the raster: each cell is a block of CELL x CELL pixels."""
    barrier = np.kron(cells, np.ones((CELL, CELL), bool))
    return np.pad(barrier, [(MARGIN[0], MARGIN[0]), (MARGIN[1], MARGIN[1])])


def corridors(cells: np.ndarray, shape: tuple[int, int]) -> list[np.ndarray]:
    """Candidate corridors: runs of at most MAX_CORRIDOR_CELLS barrier cells
    along a row or a column of the grid, with habitat on both sides. Returns
    the flat pixel indices of each."""
    candidates = []
    for transposed in (False, True):
        grid = cells.T if transposed else cells
        for line, values in enumerate(grid):
            # The margin is habitat, beyond both ends of every line.
            padded = np.concatenate([[False], values, [False]]).astype(int)
            edges = np.flatnonzero(np.diff(padded))
            for start, end in zip(edges[::2], edges[1::2]):
                if end - start > MAX_CORRIDOR_CELLS:
                    continue
                mask = np.zeros(shape, bool)
                for p in range(start, end):
                    r, c = (p, line) if transposed else (line, p)
                    y, x = MARGIN[0] + r * CELL, MARGIN[1] + c * CELL
                    mask[y : y + CELL, x : x + CELL] = True
                candidates.append(np.flatnonzero(mask))
    return candidates


def restore(barrier: np.ndarray, candidates: list[np.ndarray]):
    """Restore corridors in batches; return the elasticity raster and the mask
    of restored pixels at each iteration and in the final landscape."""
    quality = jnp.asarray(~barrier, jnp.float32)
    permeability = np.where(barrier, BARRIER_PERMEABILITY, 1.0).astype(np.float32)
    restored = np.zeros(barrier.shape, bool)
    history = []
    for iteration in range(ITERATIONS + 1):
        sensitivity = SensitivityAnalysis(
            quality_raster=quality,
            permeability_raster=jnp.asarray(permeability),
            distance=DISTANCE,
            dependency_range=DEPENDENCY_RANGE,
            batch_size=4096,  # one tile holds every block centre
            coarsening_factor=COARSENING,
        ).run("permeability")
        # Elasticity up to the factor 1 / C, which leaves the ranking unchanged.
        elasticity = permeability * np.asarray(sensitivity)
        history.append((elasticity, restored.copy()))
        if iteration == ITERATIONS:
            break
        gain = np.array([elasticity.ravel()[c].sum() for c in candidates])
        for _ in range(CORRIDORS_PER_ITERATION):
            near = (distance_transform_edt(~restored) < SPACING).ravel()
            gain[[near[c].any() for c in candidates]] = -np.inf
            if np.isfinite(gain.max()):
                restored.ravel()[candidates[np.argmax(gain)]] = True
        permeability[restored] = 1.0
    return history


def rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i : i + 2], 16) for i in (1, 3, 5)], float)


def key_frames(barrier: np.ndarray, history) -> list[np.ndarray]:
    """RGB frames at SCALE: habitat shaded by the logarithm of its elasticity,
    smoothed within habitat, in flat bands on a scale shared by all frames;
    then the title and the restored corridors."""
    # Random projections can make a few elasticities negative.
    logs = [np.log(np.maximum(elasticity, 1e-30)) for elasticity, _ in history]
    lo, hi = np.percentile(np.concatenate([log[~barrier] for log in logs]), [3, 99.5])
    h, w = barrier.shape
    i = (np.arange(h * SCALE) + 0.5) / SCALE - 0.5
    j = (np.arange(w * SCALE) + 0.5) / SCALE - 0.5
    grid = np.meshgrid(i, j, indexing="ij")
    bands = np.array([rgb(c) for c in HABITAT])

    def upscale(mask):
        return np.repeat(np.repeat(mask, SCALE, axis=0), SCALE, axis=1)

    frames = []
    for log, (_, restored) in zip(logs, history):
        habitat = (~barrier | restored).astype(float)
        smooth = gaussian_filter(np.where(habitat > 0, log, 0), SMOOTHING)
        smooth /= np.maximum(gaussian_filter(habitat, SMOOTHING), 1e-9)
        level = map_coordinates(
            (smooth - lo) / (hi - lo), grid, order=1, mode="nearest"
        )
        band = np.clip((level * len(HABITAT)).astype(int), 0, len(HABITAT) - 1)
        frame = bands[band]
        frame[upscale(barrier & ~restored)] = rgb(BARRIER)
        frame[upscale(restored)] = rgb(CORRIDOR)
        frames.append(frame)
    return frames


def save_gif(path: str, images: list[np.ndarray], durations: list[int]) -> None:
    """Save RGB frames as a GIF with one palette, so that colours do not
    flicker between frames."""
    pixels = np.concatenate([image.reshape(-1, 3) for image in images])
    median_cut = Image.fromarray(pixels[None]).quantize(256, dither=Image.Dither.NONE)
    palette = np.array(median_cut.getpalette()[:768]).reshape(-1, 3)
    gif = []
    for image in images:
        # The nearest palette colour of each distinct colour of the frame
        code = image.astype(np.int64) @ np.array([1 << 16, 1 << 8, 1])
        unique, inverse = np.unique(code, return_inverse=True)
        unique = np.stack([unique >> 16, (unique >> 8) & 255, unique & 255], axis=1)
        nearest = ((unique[:, None] - palette[None]) ** 2).sum(-1).argmin(1)
        index = nearest[inverse.ravel()].reshape(image.shape[:2]).astype(np.uint8)
        frame = Image.fromarray(index, "P")
        frame.putpalette(palette.astype(np.uint8).ravel().tolist())
        gif.append(frame)
    gif[0].save(path, save_all=True, append_images=gif[1:], duration=durations, loop=0)


def render(frames: list[np.ndarray]) -> None:
    """Write the animation, fading from one iteration to the next, and its last
    frame."""
    images, durations = [frames[0]], [FIRST_MS]
    for start, end in zip(frames, frames[1:]):
        for step in range(1, FADE_FRAMES + 1):
            images.append(start + (end - start) * step / FADE_FRAMES)
            durations.append(FADE_MS)
        durations[-1] = HOLD_MS
    durations[-1] = LAST_MS
    images = [np.round(image).astype(np.uint8) for image in images]
    save_gif("logo.gif", images, durations)
    Image.fromarray(images[-1]).save("logo.png")


if __name__ == "__main__":
    start = time.perf_counter()
    cells = glyph_cells(WORD)
    barrier = landscape(cells)
    candidates = corridors(cells, barrier.shape)
    print(f"{jax.devices()[0].device_kind}: raster {barrier.shape}")
    print(f"{len(candidates)} candidate corridors")
    history = restore(barrier, candidates)
    print(f"analyses: {time.perf_counter() - start:.1f} s")
    render(key_frames(barrier, history))
    print(f"total: {time.perf_counter() - start:.1f} s")
