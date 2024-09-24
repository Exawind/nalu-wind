.. _theory_time_discretization:

Time discretization
-------------------

Time integrators range from simple backward Euler or a second order
three state scheme, BDF2.

A general time discretization approach can be written as,

.. math::

   \int \frac{\partial \rho \phi }{\partial t} dV = \int \frac{ (\gamma_1 \rho^{n+1} \phi^{n+1} 
   + \gamma_2 \rho^{n} \phi^{n} + \gamma_3 \rho^{n-1} \phi^{n-1})} {\Delta t} dV

where :math:`\gamma_i` represent the appropriate factors for either
Backward Euler or a three-point BDF2 scheme. In both discretization
approaches, the value for density and other dofs are evaluated at the
node. As such, the time contribution is a lumped mass scheme with the
volume simply the dual volume. The topology over one loops to assemble
system is simply the node. Although CVFEM affords the use of a
consistent mass matrix, this scheme is not used at present.
