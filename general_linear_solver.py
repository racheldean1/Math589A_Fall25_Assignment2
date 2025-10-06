import numpy as np
from plu_decomposition import paqlu_decomposition_in_place  # uses your Problem 1 code

# We’ll use a fixed numerical tolerance everywhere.
TOL = 1e-6


def solve(A, b, tol: float = TOL):
    """
    Solve A x = b and return a parametric description x = N @ x_free + c.

    Returns
    -------
    N : (n, f) ndarray
        Columns form a basis for the nullspace of A (so A @ N ≈ 0).
        If there are no free variables, this will be shape (n, 0).
    c : (n,) or (n, k) ndarray
        A particular solution (free variables set to zero).
        If b is m-by-k, c is n-by-k. If b is length-m, c is length-n.

    Notes
    -----
    - This function **does not** modify the caller's A: we factor a copy.
    - The factorization is PAQ = LU with:
        * P : row permutation (actual/simulated swaps)
        * Q : column permutation (virtual order only)
        * A_copy : overwritten in-place to store L (below diag) and U (on/above diag)
    - We do forward/back substitution using those permutations.
    """
    # ----- basic checks / shapes -----
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise ValueError("A must be a 2D NumPy array.")
    if not (np.issubdtype(A.dtype, np.floating) or np.issubdtype(A.dtype, np.complexfloating)):
        raise TypeError("A must have a real or complex floating dtype.")

    m, n = A.shape

    # Make b 2D (so code works for one or many right-hand sides the same way)
    b = np.asarray(b, dtype=A.dtype)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if b.shape[0] != m:
        raise ValueError("b must have the same number of rows as A.")

    # ----- factorization (work on a copy so the caller's A isn't destroyed) -----
    A_copy = np.array(A, copy=True)
    P, Q, r = paqlu_decomposition_in_place(A_copy, tol=tol)  # your PAQ=LU (Problem 1)

    # ----- apply row permutation to b: y will hold the 'Ly = Pb' values -----
    y = b[P, :].copy()  # y starts as Pb; we'll forward-substitute in-place

    # ----- forward substitution: solve L y = Pb for the first r rows -----
    # L has unit diagonal; its sub-diagonal entries are stored in A_copy at [P[i], Q[j]] for j < i.
    for i in range(r):
        for j in range(i):  # subtract L[i,j] * y[j] from y[i]
            lij = A_copy[P[i], Q[j]]
            if lij != 0:
                y[i, :] -= lij * y[j, :]

    # ----- consistency check: rows below rank should imply 0 = y[i] -----
    if r < m:
        # If any of these entries are larger than tol, there is no solution.
        if np.max(np.abs(y[r:m, :])) > tol:
            raise ValueError("inconsistent system: A x = b has no solution within tolerance")

    # ----- back substitution: solve U z = y for the first r rows -----
    # U's entries live in A_copy at [P[i], Q[j]] for j >= i.
    z = np.zeros((r, y.shape[1]), dtype=A_copy.dtype)
    for i in range(r - 1, -1, -1):
        rhs = y[i, :].copy()
        # subtract contributions from already-solved variables
        for j in range(i + 1, r):
            rhs -= A_copy[P[i], Q[j]] * z[j, :]
        diag = A_copy[P[i], Q[i]]
        if np.abs(diag) <= tol:
            # If rhs is not ~0 here, it would be inconsistent; but that should
            # have been caught above. We'll just set zero to be safe.
            z[i, :] = 0
        else:
            z[i, :] = rhs / diag

    # ----- build a particular solution 'c' (free variables set to 0) -----
    # In the "permuted variable order", the first r variables are the pivots
    # (entries of z) and the last (n-r) are the free ones (set to 0 here).
    c_perm = np.zeros((n, y.shape[1]), dtype=A_copy.dtype)
    if r > 0:
        c_perm[:r, :] = z

    # Map back to the original column order using the inverse of Q
    invQ = np.empty_like(Q)
    invQ[Q] = np.arange(n)

    c = np.zeros_like(c_perm)
    for j_perm in range(n):
        j_orig = Q[j_perm]      # original column index that sits at position j_perm
        c[j_orig, :] = c_perm[j_perm, :]

    # ----- build a nullspace basis N (so A @ N ≈ 0) -----
    f = n - r  # number of free variables
    if f <= 0:
        # Full column rank: trivial nullspace
        N = np.zeros((n, 0), dtype=A_copy.dtype)
    else:
        # We need the r×r leading triangular block Ur and the r×f trailing block Ut from U.
        # We'll extract them by hand via P and Q.
        Ur = np.zeros((r, r), dtype=A_copy.dtype)
        Ut = np.zeros((r, f), dtype=A_copy.dtype)
        for i in range(r):
            for j in range(r):
                Ur[i, j] = A_copy[P[i], Q[j]]
            for j in range(f):
                Ut[i, j] = A_copy[P[i], Q[r + j]]

        # Each free variable gives one nullspace vector.
        # In permuted coordinates, a basis vector looks like [v ; e_k],
        # where v solves Ur v = -Ut[:, k].
        N_perm = np.zeros((n, f), dtype=A_copy.dtype)

        for k in range(f):
            # Solve Ur * v = -Ut[:, k] by simple back substitution.
            rhs = -Ut[:, k].astype(A_copy.dtype)
            v = np.zeros(r, dtype=A_copy.dtype)
            for i in range(r - 1, -1, -1):
                s = rhs
                # subtract known U parts
                for j in range(i + 1, r):
                    s -= Ur[i, j] * v[j]
                if np.abs(Ur[i, i]) <= tol:
                    v[i] = 0
                else:
                    v[i] = s / Ur[i, i]
                rhs = rhs  # (just to emphasize we're reusing the same 'rhs' variable)

            # Fill the permuted vector: first r entries are v, then a 1 in the k-th free slot
            N_perm[:r, k] = v
            N_perm[r + k, k] = 1.0

        # Map N_perm back to original column order using Q
        N = np.zeros_like(N_perm)
        for j_perm in range(n):
            j_orig = Q[j_perm]
            N[j_orig, :] = N_perm[j_perm, :]

    # ----- if caller gave a 1D b, return a 1D c to be friendly -----
    if c.shape[1] == 1:
        c = c[:, 0]

    return N, c
