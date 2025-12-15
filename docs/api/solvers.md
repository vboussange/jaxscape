Distance computations requiring a linear system solve can be accelerated using specialized solvers. The user can select any solver compatible with the Lineax library to perform these computations, and use additional custom JAXScape solvers wrapped into Lineax solvers. 


## Using `lineax` solvers

You can use any Lineax solver with `ResistanceDistance`:

```python
import lineax as lx

# Conjugate Gradient
solver = lx.CG(rtol=1e-6, atol=1e-6)
distance = ResistanceDistance(solver=solver)
```

See the [Lineax documentation](https://docs.kidger.site/lineax/) for available solvers and their options.

## `JAXScape` solvers
::: jaxscape.solvers.pyamgsolver.PyAMGSolver
    options:
      members: false

::: jaxscape.solvers.cholmodsolver.CholmodSolver
    options:
      members: false

## Advanced: `BCOOLinearOperator`

`JAXScape` provides a custom linear operator compatible with Lineax that leverages the BCOO sparse matrix format.

::: jaxscape.solvers.operator.BCOOLinearOperator
    options:
      members: false