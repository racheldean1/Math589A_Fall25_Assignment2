Math 589A, Programming Assignment 2
===================================

Problem 1. 
==========

Implement the PAQ=LU decomposition. The algorithm should perform
partial pivoting with simulated row exchanges, using only a vector P
and matrix A, which will eventually hold the L and U components of the
decomposition (PAQ=LU decomposition "in place"; the original matrix A
is destroyed in the process). The PAQ=LU decomposition performs
row and column exchanges, so that all pivot columns appear before
non-pivot columns. The column exchange is virtual, i.e.,
only the position of pivot columns is recorded in Q.


The algorithm from the book should be modified to handle a rectangular
matrix A.  Derive the algorithm by assuming that the rectangular
matrix is padded with 0 rows or columns, which makes the matrix
square. Then the original algorithm works.  However, you should never
use the non-existent entries of your matrix. Thus, the added 0 entries 
are implicit.



Problem 2.
==========

Write a solver of all linear systems Ax=b, utilizing the PA=LU decomposition
found in Problem 1. Note that the pivot columns of U correspond to basic variables,
and other columns to free variables. General idea is to write the solution
as 
                  x = N*x_free + c
				  
where N is a suitable matrix. The separate LaTeX document
[derivation.tex](derivation.tex) in this folder derives N and c.
	             


				  



