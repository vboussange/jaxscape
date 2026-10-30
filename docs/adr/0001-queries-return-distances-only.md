# Queries return distances only

Solver state follows one lifecycle for every distance and solver: `init` builds it, `reinit` produces the state of another graph, and queries read it. A carried optimization loop calls `reinit` and then a query, and needs no query that also returns the state. A `reinit` of unchanged values does nothing, so a query can call it on any state it receives: with a solver that reuses state, queries solve the graph they are given however outdated the state is, and queries on one graph share one factorization.

Spec #41 and ticket #42 asked for state-returning forms of each query instead. Those forms doubled the query methods of each distance, moved solver setup out of the custom gradient rules, and refactorized once per query, also on an unchanged graph. Queries that never touch the state are simpler still, but return silently wrong distances for an outdated state.
