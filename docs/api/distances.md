# Distance metrics

::: jaxscape.distance.AbstractDistance
    options:
      members: false

::: jaxscape.euclidean_distance.EuclideanDistance
    options:
      members: false

::: jaxscape.lcp_distance.LCPDistance
    options:
      members: false

::: jaxscape.resistance_distance.ResistanceDistance
    options:
      members: false

::: jaxscape.resistance_distance.ExactResistance
    options:
      members: false

::: jaxscape.resistance_distance.SpielmanApproximation
    options:
      members: false

::: jaxscape.free_energy_distance.FreeEnergyDistance
    options:
      members: false

## Free-energy distance

`FreeEnergyDistance(theta, solver=None)` takes a finite positive scalar inverse
temperature and computes the symmetric first-hitting
free-energy distance on a connected graph with symmetric conductance matrix
$A=(a_{ij})$. Present edges have finite positive conductance and reciprocal cost
$c_{ij}=1/a_{ij}$. Zero adjacency entries represent absent edges. Duplicate
sparse entries are added before applying the edge-cost transformation.

Use an explicitly symmetric weight function for nonuniform rasters:

```python
import jax.numpy as jnp
from jaxscape import FreeEnergyDistance, GridGraph

grid = GridGraph(
    jnp.array([[1.0, 0.8], [0.6, 1.0]]),
    fun=lambda x, y: (x + y) / 2,
)
distance = FreeEnergyDistance(theta=0.2)
values = distance(grid, nodes=jnp.array([[0, 0], [1, 1]]))
```

Rook or queen connectivity alone does not ensure symmetric conductances:
`GridGraph`'s default weight function uses the target cell value.

### Definition and limits

Let $P_{ij}=a_{ij}/\sum_k a_{ik}$ be the reference random walk, and let
$q_{st}(\pi)$ be the product of its transition probabilities along a walk that
first reaches $t$ at its final step. With total reciprocal edge cost $C(\pi)$,

$$
h_{st}=\sum_{\pi:s\to t}q_{st}(\pi)e^{-\theta C(\pi)},\qquad
\phi_{st}=-\frac{\log h_{st}}{\theta},\qquad
d_{\mathrm{FE}}(s,t)=\frac{\phi_{st}+\phi_{ts}}2.
$$

The parameter $\theta>0$ is **inverse temperature**. The free energy includes
both expected walk cost and a relative-entropy penalty against the reference
walk. Expected walk cost alone is a different quantity. The symmetrized
free energy is a metric on the supported graphs; the directed potentials
$\phi_{st}$ need not be symmetric.

For fixed conductances,

$$
\lim_{\theta\to\infty}d_{\mathrm{FE}}(s,t)
=d_{\mathrm{LCP}}(s,t; c=1/a),
$$

$$
\lim_{\theta\to0^+}d_{\mathrm{FE}}(s,t)
=\frac12\left(\sum_{i,j}a_{ij}c_{ij}\right)R_{\mathrm{eff}}(s,t).
$$

The sum includes both orientations. For a loop-free graph with $m$ undirected
edges and reciprocal costs, the second limit is $mR_{\mathrm{eff}}$.
No normalization removes this factor. These are mathematical limits; finite
precision need not resolve the calculation at arbitrarily extreme $\theta$.
See [Kivimäki, Shimbo and Saerens](https://arxiv.org/html/1212.1666) for the
free-energy definition and commute-cost normalization, and
[Françoisse et al.](https://apiacoa.org/publications/2017/francoissekivimakietal2017bag-of-path.pdf)
for the first-hitting metric construction.

### Solves and differentiation

Write $D=\operatorname{diag}(A\mathbf1)$,
$B_{ij}=a_{ij}e^{-\theta/a_{ij}}$ on present edges, and $H=D-B$.
The same symmetric positive-definite operator serves all queries. If
$Y=H^{-1}$, then $h_{st}=Y_{st}/Y_{tt}$. The implementation solves $HX=E_T$
for the coordinate-basis columns corresponding to the requested endpoints.
It uses both separately computed reciprocal entries when symmetrizing.

With an explicit sparse solver, a subset query retains sparse assembly and
solves only the selected columns. The default `solver=None` uses a dense
solve for small graphs. The usual `nodes`, `sources`, `targets`, omitted-side,
repeated-index and grid-coordinate conventions apply.

First-order reverse-mode gradients include conductances in both the reference
walk and reciprocal costs, and also include the explicit inverse-temperature
factor. The selected-inverse custom VJP reuses the saved forward solutions:
for selected-block cotangent $G$, the adjoint is $XG$. It requires no additional
linear solve and retains no solver-iteration tape. Higher-order and
forward-mode differentiation are outside this interface's guarantee.

For $N$ vertices, $M$ stored sparse entries and $K$ selected columns, solution
storage is $O(NK)$, in addition to the selected block and requested output.
The custom backward costs $O(NK^2)$ for $XG$ plus $O(MK)$ for sparse operator
gradients. All-pairs output inherently requires quadratic storage. Solver
setup, forward costs and memory depend on the chosen backend.

### Numerical range

Float32 and float64 follow ordinary JAX dtype rules. Float64 is recommended;
JAXScape neither enables it globally nor silently promotes float32 inputs.
Construct float64 arrays explicitly after enabling x64 in your application.

The killing term is evaluated using `expm1`, and distances use logarithm
differences. A private acceptance check uses a sparse block residual product,
rounding allowances and a dtype- and distance-scale-dependent threshold to
reject nonpositive or unresolved queried log arguments and inaccurate distance
estimates. It adds $O(MK)$ work and no solves. Its threshold is separate from
the solver's convergence tolerance; a converged solve can still leave tiny
queried entries unresolved. The check is a conservative floating-point
heuristic, not an unconditional accuracy certificate. Gradient accuracy is
validated separately from forward acceptance. The current private threshold for
the estimated distance error is
$16\sqrt{\epsilon_{\mathrm{mach}}}\max(1,|d|)$, where
$\epsilon_{\mathrm{mach}}$ is the dtype's machine epsilon and $d$ is the
computed distance. Independent public-interface value and gradient tests
calibrate ordinary accepted ranges; this coefficient is an implementation
detail rather than a public accuracy setting.

Self-distance and its gradient are structurally zero. Off-diagonal failures
raise an Equinox/JIT-compatible error, using normal backend error behavior.
There is no clipping, regularization, precision promotion or substitution of a
limiting distance. Try a different inverse temperature, higher precision or
tighter solver settings when the requested calculation is unresolved.
As documented by [Lineax](https://docs.kidger.site/lineax/api/linear_solve/),
errors are reliably raised on CPU; on GPU a failure may only print an error.

Full solver state depends on both $A$ and $\theta$; rebuild it when either
changes. See the [solver guide](linear_solvers.md) for initialization and
preconditioner-only reuse.
