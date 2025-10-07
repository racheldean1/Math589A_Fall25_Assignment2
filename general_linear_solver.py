import numpy as np
from plu_decomposition import paqlu_decomposition_in_place, TOL  # TOL = 1e-6

def _is_numeric_dtype(dt):
    return np.issubdtype(dt, np.floating) or np.issubdtype(dt, np.complexfloating)

def _rank_from_diagonal(A_mod, P, Q, tol):
    """Numerical rank = count of diagonal pivots |U[i,i]| > tol in permuted order."""
    m, n = A_mod.shape
    r = 0
    for i in range(min(m, n)):
        if np.abs(A_mod[P[i], Q[i]]) > tol:
            r += 1
        else:
            break
    return r

def solve(A, b, tol: float = TOL):
    """
    Parametric solver: returns (N, c) with x = N @ x_free + c.
    N is n×f (f = n - rank), c is n or n×k.
    """
    # ---- checks / shaping ----
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise ValueError("A must be a 2D NumPy array.")
    if not _is_numeric_dtype(A.dtype):
        raise TypeError("A must have a real or complex floating dtype.")
    m, n = A.shape

    b = np.asarray(b, dtype=A.dtype)
    one_rhs = False
    if b.ndim == 1:
        b = b.reshape(-1, 1)
        one_rhs = True
    if b.shape[0] != m:
        raise ValueError("b must have the same number of rows as A.")

    # ---- PAQ = LU on a copy (keep user's A intact) ----
    A_work = np.array(A, copy=True)
    P, Q, A_mod = paqlu_decomposition_in_place(A_work, tol=tol)
    r = _rank_from_diagonal(A_mod, P, Q, tol)  # <-- pivot-count rank
    f = n - r                                  # number of free variables

    # ---- Forward substitution on L y = P b (first r rows) ----
    y = b[P, :].copy()
    for i in range(r):
        for j in range(i):
            lij = A_mod[P[i], Q[j]]
            if lij != 0:
                y[i, :] -= lij * y[j, :]

    # ---- Consistency check only for rows whose U-row is ~0 ----
    if r < m:
        for i in range(r, m):
            # consider active tail of U in columns r..n-1
            u_tail = np.abs(A_mod[P[i], Q[r:n]]) if n > r else np.array([])
            u_zero_row = (u_tail.size == 0) or (np.max(u_tail) <= tol)
            if u_zero_row and np.max(np.abs(y[i, :])) > tol:
                raise ValueError("inconsistent system: A x = b has no solution within tolerance")

    # ---- Back substitution on U z = y (first r rows) ----
    z = np.zeros((r, y.shape[1]), dtype=A_mod.dtype)
    for i in range(r - 1, -1, -1):
        rhs = y[i, :].copy()
        for j in range(i + 1, r):
            rhs -= A_mod[P[i], Q[j]] * z[j, :]
        diag = A_mod[P[i], Q[i]]
        if np.abs(diag) <= tol:
            z[i, :] = 0
        else:
            z[i, :] = rhs / diag

    # ---- Particular solution c (free vars = 0) ----
    c_perm = np.zeros((n, y.shape[1]), dtype=A_mod.dtype)
    if r > 0:
        c_perm[:r, :] = z

    # map back to original column order
    c = np.zeros_like(c_perm)
    for j_perm in range(n):
        c[Q[j_perm], :] = c_perm[j_perm, :]

    # ---- Nullspace basis N (n×f) ----
    if f <= 0:
        N = np.zeros((n, 0), dtype=A_mod.dtype)        # explicit 2-D empty matrix
    else:
        # Extract Ur (r×r) and Ut (r×f) from U in permuted order
        Ur = np.zeros((r, r), dtype=A_mod.dtype)
        Ut = np.zeros((r, f), dtype=A_mod.dtype)
        for i in range(r):
            for j in range(r):
                Ur[i, j] = A_mod[P[i], Q[j]]
            for j in range(f):
                Ut[i, j] = A_mod[P[i], Q[r + j]]

        # Build N in permuted coords: columns are [v; e_k], Ur v = -Ut[:, k]
        N_perm = np.zeros((n, f), dtype=A_mod.dtype)
        for k in range(f):
            rhs = -Ut[:, k].astype(A_mod.dtype)        # length r
            v = np.zeros(r, dtype=A_mod.dtype)
            for i in range(r - 1, -1, -1):
                s = rhs[i]
                for j in range(i + 1, r):
                    s -= Ur[i, j] * v[j]
                if np.abs(Ur[i, i]) <= tol:
                    v[i] = 0
                else:
                    v[i] = s / Ur[i, i]
            N_perm[:r, k] = v
            N_perm[r + k, k] = 1.0

        # map back to original order and hard-ensure 2-D
        N = np.zeros_like(N_perm)
        for j_perm in range(n):
            N[Q[j_perm], :] = N_perm[j_perm, :]
        if N.ndim == 1:           # f == 1 edge case
            N = N.reshape(n, 1)
        elif N.shape == (n,):
            N = N.reshape(n, 1)

    # return 1-D c if user gave a 1-D b
    if one_rhs:
        c = c[:, 0]

    return N, c
