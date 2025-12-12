# Graph Classes

::: jaxscape.graph.AbstractGraph
    options:
      show_source: true
      heading_level: 2

::: jaxscape.graph.Graph
    options:
      show_source: true
      heading_level: 2

::: jaxscape.graph.GridGraph
    options:
      show_source: true
      heading_level: 2

## Contiguity Patterns

JAXScape provides predefined contiguity patterns for `GridGraph`:

### ROOK_CONTIGUITY

4-connectivity pattern (cardinal directions only):

```python
from jaxscape import ROOK_CONTIGUITY, GridGraph

grid = GridGraph(grid=permeability, neighbors=ROOK_CONTIGUITY)
```

Connects each node to its:
- North neighbor
- South neighbor
- East neighbor
- West neighbor

### QUEEN_CONTIGUITY

8-connectivity pattern (cardinal and diagonal directions):

```python
from jaxscape import QUEEN_CONTIGUITY, GridGraph

grid = GridGraph(grid=permeability, neighbors=QUEEN_CONTIGUITY)
```

Connects each node to all 8 surrounding neighbors.

