import numpy as np
from plu_decomposition import paqlu_decomposition_in_place, TOL  # use same tol

def _is_numeric_dtype(dt):
    return np.issubdtype(dt, np.floating) or np.issubdtype(dt, np.complexfloating)


def _rank_from_U(A_mod, P, Q, tol):
    """
    Determine numerical rank r by counting non-small diagonals of U
    along the first min(m, n) pivots in the *permuted* order.
    """
    m, n = A_mod.shape
    mn = min(m, n)
    r = 0
    for k in range(mn):
        if np.abs(A_mod[P[k], Q[k]]) > tol:
            r += 1
        else:
            break
    return r


def solve(A, b, tol: float = TOL):
    """
    Return a parametric solution of A x = b in the form:
        x = N @ x_free + c
    where columns of N span the nullspace and c is one particular solution.

    This uses only low-level NumPy and simple loops (no np.linalg.solve etc.).
    """
    # ----- validate inputs / shapes -----
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise ValueError("A must be a 2D NumPy array.")
    if not _is_numeric_dtype(A.dtype):
        raise TypeError("A must have a real or complex floating dtype.")

    m, n = A.shape

    b = np.asarray(b, dtype=A.dtype)
    if b.ndim == 1:
        b = b.reshape(-1, 1)  # treat as m-by-1
    if b.shape[0] != m:
        raise ValueError("b must have the same number of rows as A.")

    # ----- factorization (on a COPY so we don't mutate the caller's A) -----
    A_work = np.array(A, copy=True)
    P, Q, A_mod = paqlu_decomposition_in_place(A_work, tol=tol)

    # Recompute numerical rank from the triangular U block
    r = _rank_from_U(A_mod, P, Q, tol)

    # ----- apply row permutation to b -----
    y = b[P, :].copy()  # y will be overwritten to hold "Ly = Pb" partial results

    # ----- forward substitution for first r rows (L has unit diagonal) -----
    for i in range(r):
        for j in range(i):
            lij = A_mod[P[i], Q[j]]  # L(i,j) stored below diag
            if lij != 0:
                y[i, :] -= lij * y[j, :]

    # ----- consistency check on remaining rows (they must be ~0) -----
    if r < m and np.max(np.abs(y[r:m, :])) > tol:
        raise ValueError("inconsistent system: A x = b has no solution within tolerance")

    # ----- back substitution: solve U z = y for first r rows -----
    z = np.zeros((r, y.shape[1]), dtype=A_mod.dtype)
    for i in range(r - 1, -1, -1):
        rhs = y[i, :].copy()
        for j in range(i + 1, r):
            rhs -= A_mod[P[i], Q[j]] * z[j, :]
        diag = A_mod[P[i], Q[i]]
        if np.abs(diag) <= tol:
            z[i, :] = 0  # should not happen if r is computed correctly, but keep safe
        else:
            z[i, :] = rhs / diag

    # ----- particular solution c with free vars = 0 (in permuted order) -----
    c_perm = np.zeros((n, y.shape[1]), dtype=A_mod.dtype)
    if r > 0:
        c_perm[:r, :] = z

    # Map back to original column order using inverse of Q
    c = np.zeros_like(c_perm)
    for j_perm in range(n):
        j_orig = Q[j_perm]
        c[j_orig, :] = c_perm[j_perm, :]

    # ----- build nullspace basis N (size n-by-f, f = n - r) -----
    f = n - r
    if f <= 0:
        N = np.zeros((n, 0), dtype=A_mod.dtype)
    else:
        # Extract r×r Ur and r×f Ut out of the (permuted) U
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
            # Back substitution on Ur
            rhs = -Ut[:, k].astype(A_mod.dtype)
            v = np.zeros(r, dtype=A_mod.dtype)
            for i in range(r - 1, -1, -1):
                s = rhs
                for j in range(i + 1, r):
                    s -= Ur[i, j] * v[j]
                if np.abs(Ur[i, i]) <= tol:
                    v[i] = 0
                else:
                    v[i] = s / Ur[i, i]
                rhs = rhs  # keep style simple/explicit

            # Build basis vector in permuted coordinates: [v; e_k]
            N_perm[:r, k] = v
            N_perm[r + k, k] = 1.0

        # Map N back to the original column order using Q
        N = np.zeros_like(N_perm)
        for j_perm in range(n):
            j_orig = Q[j_perm]
            N[j_orig, :] = N_perm[j_perm, :]

    # If caller passed 1-D b, return 1-D c for friendliness
    if c.shape[1] == 1:
        c = c[:, 0]

    return N, c
