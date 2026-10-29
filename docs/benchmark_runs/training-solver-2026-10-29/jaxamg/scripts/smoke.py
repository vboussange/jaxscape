import jax, jax.numpy as jnp, numpy as np, scipy.sparse as sp
import jax.experimental.sparse as jsp
import jaxamg
print(jax.__version__, jaxamg.__version__, jax.devices())
n = 50
A = sp.diags([-1, 2.0001, -1], [-1, 0, 1], shape=(n, n), format="csr"); A.sort_indices()
Aj = jsp.BCSR((jnp.asarray(A.data), jnp.asarray(A.indices, jnp.int32), jnp.asarray(A.indptr, jnp.int32)), shape=A.shape)
b = jnp.ones(n)
x, info = jaxamg.solve(Aj, b, config={"solver": "PCG"})
print(info["iterations"], info["status"], float(jnp.linalg.norm(Aj @ x - b)))
