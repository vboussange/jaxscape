
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

The default `LCPDistance()` stops at exact convergence, retaining the $N-1$
round bound. Hard least-cost values, selection forms and Floyd–Warshall are
unchanged. Reverse mode shares adjoints equally among tight incoming edges
($d_u+1/a_{uv}=d_v$ exactly); near ties are excluded. For output cotangents $g_v$
and incoming shares $\alpha_{uv}$, the sparse backward recurrence is
$\lambda_u=g_u+\sum_v\alpha_{uv}\lambda_v$, giving
$\partial L/\partial a_{uv}=-\alpha_{uv}\lambda_v/a_{uv}^2$, summed over sources.
Ordinary AD propagates this through `GridGraph.fun` to raster/model parameters.
The implementation comments derive the rule without a forward-history tape.

- **Unique path:** ordered edges $(st,su,ut)$ with costs $(4,1,2)$ give distance
  $3$ and conductance gradient $(0,-1,-4)$, the ordinary path derivative.
- **Changed tie behavior:** costs $(2,1,1)$ give $(-2,-0.5,-0.5)$, independent of
  isolated vertices. Previously, five vertices gave $(-2.25,-0.4375,-0.4375)$
  because redundant relaxation rounds affected tie weights.
- **Local sharing:** costs $(sa,at,sb,sc,cb,bt)=(1,2,2,1,1,1)$ give three optimal
  routes through $a$, $b$, and $c,b$, with weights $1/2,1/4,1/4$, not $1/3$ each.
  Their incidence mixture is a supergradient of the concave scalar distance;
  it does not smooth the objective or guarantee improved optimization.

Support covers fixed-topology sparse directed graphs, rook/queen grids, JIT,
float32/float64, first-order `grad`/`value_and_grad`/`vjp`, and `vmap` over weights
or rasters, including mapped gradients. Topology, forward-mode JVPs and higher
orders are not differentiable within this contract. Present conductances must
be finite, positive and normal (subnormals may flush to zero), with finite
positive reciprocal costs and valid indices. Zero entries/padding are absent
and receive zero cotangent. Positive duplicate entries remain parallel edges
with separate reciprocal costs and cotangents; coalescing changes the graph.

Source-to-self distance is zero and unreachable distance is infinity. Finite
losses using reachable outputs accept arbitrary finite signed cotangents;
unreachable outputs with zero cotangent do not contaminate them. Do not form
losses with `0 * infinity`. A tight edge without strict representable distance
growth, or a finite non-source distance without a tight predecessor, triggers
an Equinox runtime error (possibly wrapped under JIT). Rescale or enable float64;
there is no tolerance repair, and precision can change the identified ties.

For $S$ sources, $N$ vertices, $M$ edges and $K_f,K_b$ forward/backward rounds,
work is $O(SM(K_f+K_b))$, worst-case $O(SNM)$; working storage is $O(S(N+M))$
plus outputs. All-pairs output remains $O(N^2)$. The
[benchmark harness and raw results](https://github.com/vboussange/jaxscape/tree/main/benchmark/lcp_acceleration)
report synchronized before/after timings and XLA memory **estimates**, not
measured peak memory. CPU results do not establish GPU or large all-pairs gains.

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
