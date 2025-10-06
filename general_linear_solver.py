import numpy as np
from plu_decomposition import paqlu_decomposition_in_place, TOL

def _is_numeric_dtype(dt):
    return np.issubdtype(dt, np.floating) or np.issubdtype(dt, np.complexfloating)

def _rank_from_U(A_mod, P, Q, tol):
    """Count non-small diags of U in the permuted order to get rank r."""
    m, n = A_mod.shape
    r = 0
    for k in range(min(m, n)):
        if np.abs(A_mod[P[k], Q[k]]) > tol:
            r += 1
        else:
            break
    return r

def solve(A, b, tol: float = TOL):
    """
    Parametric solver: returns (N, c) with x = N @ x_free + c.
    Uses only basic NumPy (no high-level solvers).
    """
    # ---- input checks ----
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise ValueError("A must be a 2D NumPy array.")
    if not _is_numeric_dtype(A.dtype):
        raise TypeError("A must have a real or complex floating dtype.")

    m, n = A.shape
    b = np.asarray(b, dtype=A.dtype)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if b.shape[0] != m:
        raise ValueError("b must have the same number of rows as A.")

    # ---- factorization on a copy (keep caller's A intact) ----
    A_work = np.array(A, copy=True)
    P, Q, A_mod = paqlu_decomposition_in_place(A_work, tol=tol)
    r = _rank_from_U(A_mod, P, Q, tol)

    # ---- forward substitution on Ly = Pb (first r rows) ----
    y = b[P, :].copy()
    for i in range(r):
        for j in range(i):
            lij = A_mod[P[i], Q[j]]
            if lij != 0:
                y[i, :] -= lij * y[j, :]

    # ---- consistency check on the remaining rows ----
    if r < m and np.max(np.abs(y[r:m, :])) > tol:
        raise ValueError("inconsistent system: A x = b has no solution within tolerance")

    # ---- back substitution on Uz = y (first r rows) ----
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

    # ---- build particular solution c (free vars set to 0) in permuted order ----
    c_perm = np.zeros((n, y.shape[1]), dtype=A_mod.dtype)
    if r > 0:
        c_perm[:r, :] = z

    # map back to original column order
    c = np.zeros_like(c_perm)
    for j_perm in range(n):
        j_orig = Q[j_perm]
        c[j_orig, :] = c_perm[j_perm, :]

    # ---- build nullspace basis N (n-by-f, f = n - r) ----
    f = n - r
    if f <= 0:
        N = np.zeros((n, 0), dtype=A_mod.dtype)
    else:
        # extract Ur (r×r) and Ut (r×f) from U in permuted order
        Ur = np.zeros((r, r), dtype=A_mod.dtype)
        Ut = np.zeros((r, f), dtype=A_mod.dtype)
        for i in range(r):
            for j in range(r):
                Ur[i, j] = A_mod[P[i], Q[j]]
            for j in range(f):
                Ut[i, j] = A_mod[P[i], Q[r + j]]

        # Solve Ur v = -Ut[:, k] by simple back-substitution (SCALAR s!)
        N_perm = np.zeros((n, f), dtype=A_mod.dtype)
        for k in range(f):
            rhs = -Ut[:, k].astype(A_mod.dtype)
            v = np.zeros(r, dtype=A_mod.dtype)
            for i in range(r - 1, -1, -1):
                s = rhs[i]                          # <-- scalar, not the whole vector
                for j in range(i + 1, r):
                    s -= Ur[i, j] * v[j]
                if np.abs(Ur[i, i]) <= tol:
                    v[i] = 0
                else:
                    v[i] = s / Ur[i, i]
            # basis vector in permuted coords: [v; e_k]
            N_perm[:r, k] = v
            N_perm[r + k, k] = 1.0

        # map N back to original column order
        N = np.zeros_like(N_perm)
        for j_perm in range(n):
            j_orig = Q[j_perm]
            N[j_orig, :] = N_perm[j_perm, :]

    # return 1D c if user gave a 1D b
    if c.shape[1] == 1:
        c = c[:, 0]

    return N, c
