# Math 589A — Programming Assignment 2

## Problem 1 — PAQ = LU decomposition

Implement an in-place PAQ = LU decomposition with partial pivoting and simulated row exchanges.

Requirements and notes:

- The algorithm must perform both row and column exchanges so that pivot *columns* are moved before non-pivot columns. Column exchanges are *virtual*: only record the column ordering (a permutation vector `Q`), do not physically permute the columns of `A`.
- Row exchanges must be handled by a permutation vector `P` (simulated row exchanges). Do not physically swap rows in `A`; instead maintain and use the permutation `P` when accessing rows. The matrix `A` itself will be overwritten in-place and will ultimately contain the `L` and `U` factors (except the unit diagonal of `L`, which is implicit).
- The implementation must work for rectangular matrices `A` (m × n). To derive the algorithm, imagine padding the matrix with zero rows or columns until it is square and then applying the usual square PAQ = LU algorithm. These added zeros are only conceptual: your code must never access entries outside the original matrix bounds — treat the padding as implicit.
- The final outputs should include:
  - The overwritten matrix `A` containing the `L` (strict lower triangle, with implicit ones on the diagonal) and `U` (upper triangle).
  - A row permutation vector `P` that encodes the simulated row exchanges.
  - A column permutation vector `Q` (or equivalent structure) that records the virtual column exchanges / pivot column ordering.
  - Any additional information needed to identify which columns are pivots (useful for Problem 2).

Implementation pointers:

- Use partial pivoting: at each step, choose the pivot row (maximum magnitude entry in the current column among available rows), update `P` accordingly (simulated swap), and apply Gaussian elimination on the active submatrix while recording column pivots in `Q`.
- When the matrix is rectangular, only iterate over the valid rows and columns; do not index beyond `m` or `n`.
- After completion, `U` will appear in the upper triangular part of the stored `A`, and `L` in the strictly lower triangular part (with diagonal ones implied).

## Problem 2 — Solver using PAQ = LU

Write a solver for linear systems `A x = b` that uses the PAQ = LU decomposition from Problem 1.

Key ideas:

- After PAQ = LU is computed, pivot columns of `U` correspond to basic variables; non-pivot columns correspond to free variables.
- The general solution can be written as
  x = N x_free + c
  where `x_free` contains the free-variable values, `N` is a matrix mapping free variables to the full solution vector, and `c` is a particular solution vector. The separate LaTeX file `derivation.tex` in this folder contains the derivation of `N` and `c`.
- Implementation outline:
  1. Compute PAQ = LU for `A`, obtaining `A` overwritten with `L` and `U`, and permutation vectors `P` and `Q`.
  2. Apply the row permutation `P` to `b` (simulated via indexing) and solve Ly = Pb by forward substitution, taking into account the implicit unit diagonal of `L`.
  3. Identify pivot columns in `U` (basic variables). Partition unknowns into basic and free sets according to `Q`.
  4. Solve U_basic x_basic = y_basic by back substitution for the basic variables; express the dependent variables in terms of free variables using the structure of `U` (this yields `N` and `c`).
  5. Reconstruct the full solution `x` in the original column order using `Q` (reverse the virtual column permutation).
- Your solver should support returning:
  - One particular solution `c` (e.g., with free variables set to zero).
  - The matrix `N` (so all solutions can be generated for arbitrary free-variable choices).
  - Optionally: a parameterization function that takes `x_free` and returns `x`.

See `derivation.tex` for the detailed algebra deriving `N` and `c`.
