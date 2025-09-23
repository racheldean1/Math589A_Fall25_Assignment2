Math 589A, Programming Assignment 2
===================================

Problem 1. 
==========

Implement the PA=LU decomposition. The algorithm should perform
partial pivoting with simulated row exchanges, using only a vector P
and matrix A, which will eventually hold the L and U components of the
decomposition (PA=LU decomposition "in place"; the original matrix A
is destroyed in the process).

The algorithm from the book should be modified to handle a rectangular
matrix A.  Derive the algorithm by assuming that the rectangular
matrix is padded with 0 rows or columns, which makes the matrix
square. Then the original algorithm works.  However, you should never
use the non-existent entries of your matrix. Thus, the added 0 entries 
are implicit.




