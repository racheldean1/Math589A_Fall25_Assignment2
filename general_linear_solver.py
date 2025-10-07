import numpy as np
from plu_decomposition import paqlu_decomposition_in_place, TOL  # TOL = 1e-6

def _is_numeric_dtype(dt):
    return np.issubdtype(dt, np.floating) or np.issubdtype(dt, np.complexfloating)

def _rank_from_diagonal(A_mod, P, Q, tol):
    """Numerical rank = number of diagonal pivots |U[i,i]| > tol (permuted order)."""
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
    Parametric solver for A x = b.
    Returns N (n×f) and c such that all solutions are x = N @ x_free + c.
    """
    # ----- input checks -----
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

    # ----- PAQ = LU on a copy (keep caller's A intact) -----
    A_work = np.array(A, copy=True)
    P, Q, A_mod = paqlu_decomposition_in_place(A_work, tol=tol)
    r = _rank_from_diagonal(A_mod, P, Q, tol)
    f = n - r  # nullity (number of free variables)

    # ----- Forward substitution: L y = P b (only first r rows are used) -----
    y = b[P, :].copy()
    for i in range(r):
        for j in range(i):
            lij = A_mod[P[i], Q[j]]
            if lij != 0:
                y[i, :] -= lij * y[j, :]

    # IMPORTANT:
    # Do NOT reject based on y[r:] — in rectangular systems, the bottom rows can
    # be satisfied by free variables. We only solve the first r pivot equations.

    # ----- Back substitution: U z = y (first r rows) -----
    z = np.zeros((r, y.shape[1]), dtype=A_mod.dtype)
    for i in range(r - 1, -1, -1):
        rhs = y[i, :].copy()
        for j in range(i + 1, r):
            rhs -= A_mod[P[i], Q[j]] * z[j, :]
        diag = A_mod[P[i], Q[i]]
        if np.abs(diag) <= tol:
            # With r defined from the diag, this shouldn't happen;
            # keep safe and set zero.
            z[i, :] = 0
        else:
            z[i, :] = rhs / diag

    # ----- Particular solution c (free vars set to zero) -----
    c_perm = np.zeros((n, y.shape[1]), dtype=A_mod.dtype)
    if r > 0:
        c_perm[:r, :] = z

    # Map c back to original column order
    c = np.zeros_like(c_perm)
    for j_perm in range(n):
        c[Q[j_perm], :] = c_perm[j_perm, :]

    # ----- Nullspace basis N (n×f) -----
    if f <= 0:
        N = np.zeros((n, 0), dtype=A_mod.dtype)  # explicit 2-D empty matrix
    else:
        # Extract Ur (r×r) and Ut (r×f) from the permuted U block
        Ur = np.zeros((r, r), dtype=A_mod.dtype)
        Ut = np.zeros((r, f), dtype=A_mod.dtype)
        for i in range(r):
            for j in range(r):
                Ur[i, j] = A_mod[P[i], Q[j]]
            for j in range(f):
                Ut[i, j] = A_mod[P[i], Q[r + j]]

        # Solve Ur v = -Ut[:, k] for each free variable k
        N_perm = np.zeros((n, f), dtype=A_mod.dtype)
        for k in range(f):
            rhs = -Ut[:, k].astype(A_mod.dtype)  # length r
            v = np.zeros(r, dtype=A_mod.dtype)
            for i in range(r - 1, -1, -1):
                s = rhs[i]
                for j in range(i + 1, r):
                    s -= Ur[i, j] * v[j]
                if np.abs(Ur[i, i]) <= tol:
                    v[i] = 0
                else:
                    v[i] = s / Ur[i, i]
            # basis vector in permuted coordinates: [v; e_k]
            N_perm[:r, k] = v
            N_perm[r + k, k] = 1.0

        # Map N to original column order and ensure 2-D
        N = np.zeros_like(N_perm)
        for j_perm in range(n):
            N[Q[j_perm], :] = N_perm[j_perm, :]
        if N.ndim == 1:       # f==1 edge case
            N = N.reshape(n, 1)

    # Return 1-D c if user passed 1-D b
    if one_rhs:
        c = c[:, 0]

    return N, c
