import numpy as np

# Fixed tolerance per course guidance
TOL = 1e-6

def _is_numeric_dtype(dt):
    return np.issubdtype(dt, np.floating) or np.issubdtype(dt, np.complexfloating)

def _paqlu_core_in_place(A, tol=TOL):
    """Internal PAQ=LU that overwrites A and returns P,Q,r."""
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise ValueError("A must be a 2D NumPy ndarray.")
    if not _is_numeric_dtype(A.dtype):
        raise TypeError("A must have a real or complex floating dtype.")

    m, n = A.shape
    P = np.arange(m, dtype=int)   # row permutation (simulated swaps)
    Q = np.arange(n, dtype=int)   # column permutation (virtual order)
    r = 0                         # numerical rank

    while r < m and r < n:
        # pick column whose best pivot (by abs value) is largest
        best_col = -1
        best_row = -1
        best_val = 0.0
        for j in range(r, n):
            col = Q[j]
            abs_col = np.abs(A[P[r:m], col])
            if abs_col.size == 0:
                continue
            i_local = int(np.argmax(abs_col))
            val = float(abs_col[i_local])
            if val > best_val:
                best_val = val
                best_col = j
                best_row = r + i_local

        if best_col < 0 or best_val <= tol:
            break  # rank revealed

        # virtual column move + simulated row swap
        if best_col != r:
            Q[r], Q[best_col] = Q[best_col], Q[r]
        if best_row != r:
            P[r], P[best_row] = P[best_row], P[r]

        piv = A[P[r], Q[r]]
        if np.abs(piv) <= tol:
            break

        # eliminate below pivot, store multipliers in strict lower triangle
        for i in range(r + 1, m):
            ri = P[i]
            A[ri, Q[r]] = A[ri, Q[r]] / piv
            lij = A[ri, Q[r]]
            if lij != 0:
                base = P[r]
                for j in range(r + 1, n):
                    A[ri, Q[j]] -= lij * A[base, Q[j]]

        r += 1

    return P, Q, r

def paqlu_decomposition_in_place(A, tol=TOL):
    """
    PUBLIC interface the autograder calls:
        P, Q, A_mod = paqlu_decomposition_in_place(A0)
    """
    if not isinstance(A, np.ndarray):
        A = np.asarray(A, dtype=float)
    P, Q, _r = _paqlu_core_in_place(A, tol=tol)
    return P, Q, A
