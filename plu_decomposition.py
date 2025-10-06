import numpy as np

# Fixed tolerance everywhere per your request / class guidance
TOL = 1e-6


def _is_numeric_dtype(dt):
    return np.issubdtype(dt, np.floating) or np.issubdtype(dt, np.complexfloating)


def _paqlu_core_in_place(A, tol=TOL):
    """
    Internal helper that actually performs PAQ=LU.
    It overwrites A in-place with L (strictly below diag, ones implicit)
    and U (on/above diag in the *virtual* column order tracked by Q).
    Returns (P, Q, r).  We keep this private so the public wrapper can
    expose the grader's required signature without losing rank info.
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise ValueError("A must be a 2D NumPy ndarray.")
    if not _is_numeric_dtype(A.dtype):
        raise TypeError("A must have a real or complex floating dtype.")

    m, n = A.shape
    P = np.arange(m, dtype=int)  # row permutation (simulated swaps)
    Q = np.arange(n, dtype=int)  # column permutation (virtual order)

    r = 0  # numerical rank = number of successful pivots

    # Standard Gaussian elimination with:
    #  - partial row pivoting,
    #  - "partial-like" column choice: pick the column whose best pivot is largest.
    while r < m and r < n:
        best_col = -1
        best_row = -1
        best_val = 0.0

        # Scan candidate columns j >= r
        for j in range(r, n):
            col = Q[j]
            # look only in active rows i >= r
            abs_col = np.abs(A[P[r:m], col])
            if abs_col.size == 0:
                continue
            i_local = int(np.argmax(abs_col))  # index within r..m-1
            val = float(abs_col[i_local])
            if val > best_val:
                best_val = val
                best_col = j
                best_row = r + i_local

        # Stop if no good pivot
        if best_col < 0 or best_val <= tol:
            break

        # Virtually move chosen column into position r
        if best_col != r:
            Q[r], Q[best_col] = Q[best_col], Q[r]

        # Simulate row swap by updating P
        if best_row != r:
            P[r], P[best_row] = P[best_row], P[r]

        piv = A[P[r], Q[r]]
        if np.abs(piv) <= tol:
            break  # treat as zero -> done

        # Eliminate entries below pivot in the current column
        for i in range(r + 1, m):
            ri = P[i]
            # Store multiplier in strict lower triangle
            A[ri, Q[r]] = A[ri, Q[r]] / piv
            lij = A[ri, Q[r]]
            if lij != 0:
                base = P[r]
                # Update trailing columns (in the virtual order)
                for j in range(r + 1, n):
                    A[ri, Q[j]] -= lij * A[base, Q[j]]

        r += 1

    return P, Q, r


def paqlu_decomposition_in_place(A, tol=TOL):
    """
    PUBLIC interface required by the autograder.
    It must be callable as:
        P, Q, A_mod = paqlu_decomposition_in_place(A0)

    We do the factorization in-place on the passed array and
    return P, Q, and that same (now modified) array.
    """
    if not isinstance(A, np.ndarray):
        A = np.asarray(A, dtype=float)

    # Work IN PLACE on the very same array object (as assignment asks)
    P, Q, _r = _paqlu_core_in_place(A, tol=tol)
    # Return P, Q, and the modified A (which holds L and U)
    return P, Q, A
