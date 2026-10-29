"""Variant E recomputed fully in JAX: plain aggregation with frozen aggregates.

Host, once per grid shape: PyAMG builds the unsmoothed-aggregation hierarchy
(`smooth=None, improve_candidates=None`, aggregates predefined from the uniform
SA hierarchy) on the uniform graph. Its tentative prolongators T depend only on
the aggregates (B = ones), so they are frozen. We precompute, per level, the
map from every stored entry (i, j) of A_l to its Galerkin slot
(agg[i], agg[j]) in A_{l+1} and the weight t_i t_j.

Device, every refresh: A_{l+1}.data = segment_sum(t_i t_j A_l.data) and
Dinv_l = 1 / diag(A_l); the coarsest pseudo-inverse is recomputed. All output
shapes equal the template's, so a compiled solve is reused.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

import common as cm


def _keys(indices, ncols):
    return indices[:, 0].astype(np.int64) * ncols + indices[:, 1].astype(np.int64)


def build_plan(L_template: BCOO, ml_E) -> tuple:
    """Host-side symbolic setup from the template hierarchy (uniform graph)."""
    levels = ml_E.levels
    n0 = levels[0].A.shape[0]
    # Level-0: map JAXScape BCOO entries (duplicates + OOB padding) to slots.
    idx_L = np.asarray(L_template.indices)
    valid = (idx_L[:, 0] < n0) & (idx_L[:, 1] < n0)
    k_tmpl0 = _keys(np.asarray(levels[0].A.indices), n0)
    assert np.all(np.diff(k_tmpl0) > 0), "template level-0 indices must be sorted/unique"
    pos = np.searchsorted(k_tmpl0, _keys(idx_L, n0))
    pos = np.minimum(pos, k_tmpl0.size - 1)
    ok = valid & (k_tmpl0[pos] == _keys(idx_L, n0))
    assert np.all(ok == valid), "a fine entry is missing from the template pattern"
    seg0 = np.where(valid, pos, k_tmpl0.size).astype(np.int32)

    segs, weights, diag_pos = [], [], []
    for l, lvl in enumerate(levels):
        idx = np.asarray(lvl.A.indices)
        nl = lvl.A.shape[0]
        d = np.full(nl, -1, np.int64)
        on_diag = idx[:, 0] == idx[:, 1]
        d[idx[on_diag, 0]] = np.nonzero(on_diag)[0]
        assert np.all(d >= 0)
        diag_pos.append(jnp.asarray(d.astype(np.int32)))
        if l == len(levels) - 1:
            break
        # P = T (unsmoothed): exactly one entry per row.
        P_idx = np.asarray(lvl.P.indices)
        P_dat = np.asarray(lvl.P.data)
        assert P_idx.shape[0] == nl and np.all(np.diff(P_idx[:, 0]) == 1)
        agg = P_idx[np.argsort(P_idx[:, 0]), 1]
        t = P_dat[np.argsort(P_idx[:, 0])]
        nc = levels[l + 1].A.shape[0]
        # PyAMG's Galerkin product leaves CSR column indices unsorted.
        k_c = _keys(np.asarray(levels[l + 1].A.indices), nc)
        order = np.argsort(k_c)
        kk = agg[idx[:, 0]].astype(np.int64) * nc + agg[idx[:, 1]]
        q = np.minimum(np.searchsorted(k_c[order], kk), k_c.size - 1)
        p = order[q]
        assert np.all(k_c[p] == kk), "coarse pattern incomplete"
        segs.append(jnp.asarray(p.astype(np.int32)))
        weights.append(jnp.asarray(t[idx[:, 0]] * t[idx[:, 1]]))
    nse = tuple(int(lvl.A.nse) for lvl in levels)
    return jnp.asarray(seg0), tuple(segs), tuple(weights), tuple(diag_pos), nse


def make_refresh(nse):
    @jax.jit
    def refresh(L_data, seg0, segs, weights, diag_pos, template):
        ml = jax.tree_util.tree_map(lambda x: x, template)
        a = jax.ops.segment_sum(L_data, seg0, num_segments=nse[0])
        for l, lvl in enumerate(ml.levels):
            lvl.A = BCOO((a, lvl.A.indices), shape=lvl.A.shape,
                         indices_sorted=lvl.A.indices_sorted,
                         unique_indices=lvl.A.unique_indices)
            lvl.Dinv = 1.0 / a[diag_pos[l]]
            if l < len(ml.levels) - 1:
                a = jax.ops.segment_sum(a * weights[l], segs[l], num_segments=nse[l + 1])
        coarse = ml.levels[-1]
        coarse.A_dense = coarse.A.todense()
        coarse.A_inv = jnp.linalg.pinv(coarse.A_dense)
        return ml

    return refresh
