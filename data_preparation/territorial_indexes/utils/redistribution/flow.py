# SPDX-License-Identifier: Apache-2.0
"""Min-cost redistribution on a node graph (ported from the notebook, same maths).

Move the load above the balanced saturation towards the nodes with room, at a cost that grows with distance; whatever
cannot be absorbed within `d_max_km` is reported as 'unabsorbed' (leaving it where it is is cheaper than a longer move).
The parameters come from settings.yaml (`redistribution.transport_cost`), nothing is fixed here.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


class Graph:
    """Nodes with centroids in km. Arcs: every ordered pair closer than `d_max_km`.

    cost of an arc = (d / d_ref_km) ** alpha, multiplied by `mult[i, j]` (type-affinity: >= 1 for dissimilar nodes).
    d is floored at `d_min_km` (centroid distances of neighbouring nodes).
    """

    def __init__(self, xy_km, mult=None, *, alpha, d_ref_km, d_max_km, d_min_km):
        d = np.hypot(xy_km[:, None, 0] - xy_km[None, :, 0], xy_km[:, None, 1] - xy_km[None, :, 1])
        d = np.maximum(d, d_min_km)
        np.fill_diagonal(d, np.inf)
        self.n = len(xy_km)
        self.d = d
        ai, aj = np.where(d <= d_max_km)
        self.ai, self.aj = ai, aj
        self.c = (d[ai, aj] / d_ref_km) ** alpha
        if mult is not None:
            self.c = self.c * mult[ai, aj]
        self.c_pen = (d_max_km / d_ref_km) ** alpha  # cost of leaving one unit unabsorbed

    def solve(self, P, C):
        """P, C: load and ceiling per node. Returns a dict of per-node arrays (+ the flows on the arcs)."""
        n = self.n
        valid = ~(np.isnan(P) | np.isnan(C))  # a node with no data that day is left out of the day's problem
        P = np.where(valid, P, 0.0)
        C = np.where(valid, C, 0.0)
        ubar = P.sum() / C.sum()  # the balanced saturation: the same for every node
        S = np.maximum(0.0, P - ubar * C)  # supply: excess over the balanced load
        R = np.maximum(0.0, ubar * C - P)  # room: what a node can still take
        m = (S[self.ai] > 1e-9) & (R[self.aj] > 1e-9)
        out = np.zeros(n)
        inn = np.zeros(n)
        cost_o = np.zeros(n)
        cost_i = np.zeros(n)
        arcs = (np.array([], int), np.array([], int), np.array([]))
        if m.any():
            ai, aj, c = self.ai[m], self.aj[m], self.c[m]
            k = len(c)
            A = coo_matrix(
                (np.ones(2 * k), (np.r_[ai, n + aj], np.r_[np.arange(k), np.arange(k)])), shape=(2 * n, k)
            ).tocsr()
            res = linprog(c - self.c_pen, A_ub=A, b_ub=np.r_[S, R], bounds=(0, None), method="highs")
            f = res.x
            arcs = (ai, aj, f)
            np.add.at(out, ai, f)
            np.add.at(inn, aj, f)
            np.add.at(cost_o, ai, f * c)
            np.add.at(cost_i, aj, f * c)

        def nan(x):
            return np.where(valid, x, np.nan)

        return dict(
            ubar=ubar,
            S=nan(S),
            R=nan(R),
            out=nan(out),
            inn=nan(inn),
            cost=nan(cost_o + cost_i),
            unabs=nan((S - out) + (R - inn)),
            unabs_sup=nan(S - out),
            arcs=arcs,
        )
