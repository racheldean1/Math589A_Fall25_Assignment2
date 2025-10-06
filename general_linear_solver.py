import numpy as np

def _is_numeric_dtype(dt):
    return np.issubdtype(dt, np.floating) or np.issubdtype(dt, np.complexfloating)

def _machine_eps(dtype):
    # Works for real or complex (uses float eps for underlying real)
    base = np.float64 if np.issubdtype(dtype, np.complexfloating) else dtype
    return np.finfo(np.dtype(base)).eps

def _infer_default_tol(A):
    # Recommended default: tol = max(m,n) * eps * max_abs
    m, n = A.shape
    eps = _machine_eps(A.dtype)
    max_abs = np.max(np.abs(A)) if A.size else 0.0
    return max(m, n) * eps * max_abs

def paqlu_decomposition_in_place(A, tol=None):
    """
    Compute an in-place PAQ = LU factorization of an m-by-n matrix A
    with partial row pivoting and simulated column pivoting.

    Returns
    -------
    P : (m,) int ndarray
        Row permutation vector (simulated row swaps). Uses 0-based indexing.
        Interpreted as: permuted row i of PAQ is original row P[i].
    Q : (n,) int ndarray
        Column permutation vector (virtual column order). Uses 0-based indexing.
        Interpreted as: permuted column j of PAQ is original column Q[j].
    r : int
        Numerical rank (# of successful pivots).
    Notes
    -----
    - A is overwritten in place to hold L (strict lower triangle, unit diagonal
      implicit) and U (upper triangle in the permuted column order).
    - No physical column swaps are performed; we only update Q and always
      access columns via Q.
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise ValueError("A must be a 2D NumPy ndarray.")
    if not _is_numeric_dtype(A.dtype):
        raise TypeError("A must have a real or complex floating dtype.")
    m, n = A.shape
    P = np.arange(m, dtype=int)
    Q = np.arange(n, dtype=int)

    if tol is None:
        tol = _infer_default_tol(A)

    r = 0
    # Perform Gaussian elimination over the active submatrix
    while r < m and r < n:
        # --- Choose pivot column (virtual) and pivot row (actual) ---
        # For each candidate column j>=r, pick the row with max |A[row, Q[j]]| over rows i>=r.
        # Then select the column whose best pivot is largest in magnitude.
        best_col = -1
        best_row = -1
        best_val = 0.0

        # Scan candidate columns
        for j in range(r, n):
            col = Q[j]
            # magnitudes in the active rows
            if r < m:
                abs_col = np.abs(A[P[r:m], col])
                # local best in this column
                k_local = np.argmax(abs_col) if abs_col.size else 0
                val = abs_col[k_local] if abs_col.size else 0.0
                if val > best_val:
                    best_val = val
                    best_col = j
                    best_row = r + k_local

        # If no sufficiently large pivot -> done (rank revealed)
        if best_val <= tol or best_col < 0:
            break

        # Update the virtual column order: move chosen column into position r
        if best_col != r:
            Q[r], Q[best_col] = Q[best_col], Q[r]

        # Update the (simulated) row permutation: move chosen row into position r
        if best_row != r:
            P[r], P[best_row] = P[best_row], P[r]

        # Current pivot
        piv = A[P[r], Q[r]]

        # If pivot is numerically too small, stop
        if np.abs(piv) <= tol:
            break

        # --- Elimination on rows below r, across remaining columns (accessed by Q) ---
        # Compute multipliers and update trailing block
        for i in range(r + 1, m):
            row_i = P[i]
            # Multiplier stored in strict lower triangle (column Q[r])
            A[row_i, Q[r]] = A[row_i, Q[r]] / piv
            lij = A[row_i, Q[r]]
            if lij != 0:
                # Update remaining columns j = r+1 .. n-1 (virtual order)
                # A[row_i, Q[j]] -= lij * A[P[r], Q[j]]
                base_row = P[r]
                for j in range(r + 1, n):
                    A[row_i, Q[j]] -= lij * A[base_row, Q[j]]

        r += 1

    return P, Q, r  # A is overwritten in place with L (below diag) and U (on/above)

def solve(A, b, tol=None):
    """
    Solve A x = b in parametric form x = N @ x_free + c.
    Returns
    -------
    N : (n, f) ndarray
        Nullspace basis (homogeneous solutions). If full column rank (f=0),
        returns shape (n,0).
    c : (n,) or (n,k) ndarray
        A particular solution. If b has multiple RHS (m-by-k), c is (n,k).
    Raises
    ------
    ValueError if the system is inconsistent within tolerance.
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise ValueError("A must be a 2D NumPy ndarray.")
    if not _is_numeric_dtype(A.dtype):
        raise TypeError("A must have a real or complex floating dtype.")
    b = np.asarray(b, dtype=A.dtype)
    if b.ndim == 1:
        b = b.reshape(-1, 1)  # treat as m-by-1
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError("b must have length m (same number of rows as A).")

    # Work on a copy of A so the caller's A isn't destroyed by the factorization
    A_work = np.array(A, copy=True)
    if tol is None:
        tol = _infer_default_tol(A_work)

    # Factorization
    P, Q, r = paqlu_decomposition_in_place(A_work, tol=tol)

    # Apply row permutation to b (simulated)
    bp = b[P, :].copy()

    # Forward substitution: L y = bp for first r rows (unit diagonal L)
    # y overwrites first r rows of bp
    for i in range(r):
        # y[i] -= sum_{j<i} L[i,j] * y[j], where L[i,j] lives at A_work[P[i], Q[j]]
        for j in range(i):
            bp[i, :] -= A_work[P[i], Q[j]] * bp[j, :]

    # Consistency check on the remaining rows (zero rows of U). They enforce 0 = y[i]
    if r < m:
        # If any ||bp[i,:]|| > tol, inconsistent system
        resid = np.max(np.abs(bp[r:m, :])) if (m - r) > 0 else 0.0
        if resid > tol:
            raise ValueError("inconsistent system: A x = b has no solution within tolerance")

    # Back substitution: U z = y for the first r equations, in the permuted column order
    z = np.zeros((r, bp.shape[1]), dtype=A_work.dtype)
    for i in range(r - 1, -1, -1):
        rhs = bp[i, :].copy()
        # subtract U[i, j]*z[j] for j>i (U entries on A_work[P[i], Q[j]])
        for j in range(i + 1, r):
            rhs -= A_work[P[i], Q[j]] * z[j, :]
        diag = A_work[P[i], Q[i]]
        if np.abs(diag) <= tol:
            # If y[i] also ~ 0, treat as free (already handled by rank detection); otherwise inconsistent
            if np.max(np.abs(rhs)) > tol:
                raise ValueError("inconsistent system during back substitution")
            z[i, :] = 0.0
        else:
            z[i, :] = rhs / diag

    # Build a particular solution c in the original (unpermuted) column order
    c_perm = np.zeros((n, bp.shape[1]), dtype=A_work.dtype)  # variables in permuted order
    if r > 0:
        c_perm[:r, :] = z
    # free part is zero here (particular solution)
    # Map back: x = inverse_permute_columns(c_perm, Q)
    invQ = np.empty_like(Q)
    invQ[Q] = np.arange(n)
    c = np.zeros((n, bp.shape[1]), dtype=A_work.dtype)
    for j_perm in range(n):
        j_orig = Q[j_perm]
        c[j_orig, :] = c_perm[j_perm, :]

    # Build nullspace basis N (n-by-f)
    f = n - r
    if f <= 0:
        N = np.zeros((n, 0), dtype=A_work.dtype)
    else:
        # Extract the r×r upper-triangular block Ur and the trailing r×f block Ut
        Ur = np.empty((r, r), dtype=A_work.dtype)
        Ut = np.empty((r, f), dtype=A_work.dtype)
        for i in range(r):
            for j in range(r):
                Ur[i, j] = A_work[P[i], Q[j]]
            for j in range(f):
                Ut[i, j] = A_work[P[i], Q[r + j]]

        # For each free variable e_k (in permuted coordinates), solve Ur * v = -Ut[:, k]
        N_perm = np.zeros((n, f), dtype=A_work.dtype)
        # Solve triangular systems by back-substitution (since Ur is upper triangular)
        for k in range(f):
            rhs = -Ut[:, [k]]  # column vector
            v = np.zeros((r, 1), dtype=A_work.dtype)
            # back substitution on Ur v = rhs
            for i in range(r - 1, -1, -1):
                s = rhs[i, 0]
                for j in range(i + 1, r):
                    s -= Ur[i, j] * v[j, 0]
                if np.abs(Ur[i, i]) <= tol:
                    # Treat as zero (rank deficiency already encoded in r)
                    v[i, 0] = 0.0
                else:
                    v[i, 0] = s / Ur[i, i]
            # Assemble basis vector in permuted order: [v; e_k]
            N_perm[:r, k] = v[:, 0]
            N_perm[r + k, k] = 1.0

        # Map each basis vector back via inverse column permutation
        N = np.zeros((n, f), dtype=A_work.dtype)
        for j_perm in range(n):
            j_orig = Q[j_perm]
            N[j_orig, :] = N_perm[j_perm, :]

    # If the caller provided a 1-D b, return 1-D c
    c = c[:, 0] if c.shape[1] == 1 else c
    return N, c
