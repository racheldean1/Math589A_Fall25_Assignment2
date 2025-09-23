def paqlu_decomposition_in_place(A):
    # A given rectangular matrix.
    # Constructs matrix decomposition PAQ=LU in place, so that
    # P and Q are permuation matrices corresponding to row and column exchanges.
    # The purpose of column exchanges (simulated) is to place pivot columns
    # before non-pivot columns.
    # Upon return, A must contain the PAQ=LU decomposition (L and U portion).
    # Thus, the calculation must be performed "in place" style.
    # Note: P, Q must be vectors, not 2D arrays
    return P, Q, A


def solve(A, b):
    # Return matrix N and vector c such that
    # x = N@xfree + c
    # is a solution of A@x=b for every vector  xfree (free variables)
    return N, c


