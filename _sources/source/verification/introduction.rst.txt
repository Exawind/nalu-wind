Introduction
------------

The methodology used to evaluate the accuracy of each proposed
scheme will be the method of manufactured solutions. The objective of code 
verification is to reveal coding mistakes that affect the order 
of accuracy and to determine if the governing discretized equations are being solved correctly.
Quite often, the process of verification reveals algorithmic issues that would otherwise remain 
unknown.

In practice, a variety of comparison techniques exist for verification. For example, 
benchmark and code-to-code comparison are not considered rigorous due to the errors
that exist in other code solutions, such as from discretization and iteration. Analytic 
solutions and the method of manufactured solutions remain the most powerful methods for 
code verification, since they provide a means to obtain quantitative error estimations in 
space and time.

Roache has made the distinction between code verification and calculation 
verification, where calculation verification involves grid refinement required for every 
problem solution to assess the magnitude, not order, of the discretization error. Discretization
error, distinguished from modeling and iteration errors, is defined as the difference between
the exact solution to the continuum governing equations and the solution to the algebraic 
systems representation due to discretization of the continuum equations. The order of accuracy
can be determined by comparing the discretization error on successively refined grids. Thus, it
is desirable to have an exact solution for comparision to determine the discretization errors.
