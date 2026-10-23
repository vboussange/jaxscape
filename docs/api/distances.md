
::: jaxscape.distance.AbstractDistance
    options:
      members: false

::: jaxscape.euclidean_distance.EuclideanDistance
    options:
      members: false

::: jaxscape.lcp_distance.LCPDistance
    options:
      members: false

### Bellman–Ford distances and derivatives

The default `LCPDistance()` stops Bellman–Ford when distances are unchanged,
with the full $N-1$ round bound retained for long paths. It preserves the hard
least-cost objective; `algorithm="floyd-warshall"` is unchanged. Existing
`sources`, `targets`, `nodes`, coordinate and all-pairs selections still apply.

Its first-order reverse derivative shares each vertex's adjoint equally among
all tight incoming edges, those satisfying $d_u+1/a_{uv}=d_v$ exactly in the
working dtype. For a unique shortest path this is the usual path-incidence
derivative, followed by $\partial(1/a)/\partial a=-1/a^2$. For example, ordered
edges $(st,su,ut)$ with costs $(4,1,2)$ have distance $3$ and conductance
gradient $(0,-1,-4)$.

!!! note "Tie gradients intentionally changed"

    Previously, autodiff through every relaxation round made tie weights depend
    on redundant rounds and even isolated vertices. With costs $(2,1,1)$ on
    $(st,su,ut)$, the new rule gives conductance gradient $(-2,-0.5,-0.5)$;
    the old rule with five vertices gave $(-2.25,-0.4375,-0.4375)$. Both select
    valid generalized derivatives of the scalar distance at this tie.

Sharing is uniform among incoming edges, not complete paths. For equal-cost
routes $s\to a\to t$, $s\to b\to t$, and $s\to c\to b\to t$ with edge costs
$(sa,at,sb,sc,cb,bt)=(1,2,2,1,1,1)$, the route weights are $1/2,1/4,1/4$.
The selected edge-cost derivative is a convex combination of optimal path
incidences (a supergradient of the concave distance). It does not smooth the
objective or promise improved optimization. Near ties are not exact ties;
float32 and float64 can choose different optimal edges.

For output cotangents $g_v$, let $\alpha_{uv}$ be the reciprocal number of tight
incoming edges at $v$, or zero off those edges. Sparse reverse accumulation
solves $\lambda_u=g_u+\sum_v\alpha_{uv}\lambda_v$ and returns
$\partial L/\partial a_{uv}=-\alpha_{uv}\lambda_v/a_{uv}^2$, summed over
sources. Graph construction remains differentiable, so these sensitivities
propagate through `GridGraph.fun` into raster/model parameters. The implementation
comments derive this recurrence; no forward relaxation history is differentiated.

- Supports fixed-topology sparse directed `Graph`, rook/queen `GridGraph`, JIT,
  float32/float64, first-order `grad`, `value_and_grad`, `vjp`, and tested `vmap`
  compositions over weights/rasters and their gradients. Integer topology is
  not differentiable; forward-mode JVPs and higher derivatives are unsupported.
- Present conductances must be finite and positive with finite positive
  representable reciprocal costs and valid indices. Zero conductance and sparse
  padding mean absence and receive zero cotangent. Positive duplicate entries
  are separate parallel edges with separate reciprocal costs and cotangents;
  coalescing them changes the graph and can change tie weights.
- Source-to-self distance is zero; unreachable distance is infinity. Gradients
  are guaranteed for finite losses depending only on reachable outputs, including
  arbitrary finite signed cotangents. Unreachable outputs with zero cotangent
  do not contaminate them; `0 * infinity` is not a safe way to form a finite loss.
- If a positive cost rounds away and a reachable tight edge fails strict distance
  growth, the backend rejects the calculation. Missing tight predecessors are
  errors too. Rescale costs or enable float64; no tolerance-based repair applies.
  Runtime checks use Equinox and errors under JIT may be wrapped by JAX.

With $S$ sources, $M$ edges and $K_f,K_b$ actual forward/backward rounds, work is
$O(SM(K_f+K_b))$, worst-case $O(SNM)$, with $O(S(N+M))$ working storage plus
outputs. All-pairs output still costs $O(N^2)$. See the
[benchmark harness and raw results](https://github.com/vboussange/jaxscape/tree/main/benchmark/lcp_acceleration)
for synchronized before/after measurements. XLA memory estimates are not
measured peak memory; CPU results do not establish GPU or large all-pairs gains.

::: jaxscape.resistance_distance.ResistanceDistance
    options:
      members: false

::: jaxscape.resistance_distance.ExactResistance
    options:
      members: false

::: jaxscape.resistance_distance.SpielmanApproximation
    options:
      members: false

::: jaxscape.rsp_distance.RSPDistance
    options:
      members: false
