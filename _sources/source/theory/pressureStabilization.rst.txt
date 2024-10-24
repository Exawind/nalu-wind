.. _theory_pressure_stabilization:

Pressure Stabilization
----------------------

A number of papers describing the pressure stabilization approach that
Nalu-Wind uses are in the open literature,
Domino, :cite:`Domino:2006`, :cite:`Domino:2008`, :cite:`Domino:2014`. Nalu-Wind
supports an incremental fourth order approximate projection scheme with
time step scaling. By scaling, it is implied that a time scale based on
either the physical time step or a combined elemental advection and
diffusion time scale based on element length along with advection and
diffusional parameters. An alternative to the approaximate projection
concept is to view the method as a variational multiscale (VMS) method
wherebye the momentum residual augments the continuity equation. This
allows for a diagonal entry for the pressure degree of freedom.

Here, the fine-scale momentum residual is written in terms of a
projected momentum residual evaluated at the Gauss point,

.. math::
   :label: fine-scale-momentum

   \mathbf{R}(u_i) = (\frac{\partial p} {\partial x_j} - G_j p ).


The above equation is derived simply by writing a fine-scale momentum
equation at the Gauss-points and using the nodal projected residual to
reconstruct the individual terms. Therefore, the continuity equation
solved, using the VMS-based projected momentum residual, is

.. math::

   \int \frac{\partial \bar{\rho}} {\partial t}\, dV
   + \int \left( \bar{\rho} \hat{u}_i + \tau G_i \bar{P} \right) n_i\, dS
     = \int \tau \frac{\partial \bar{P}}{\partial x_i} n_i\, dS.

Above, :math:`G_i \bar{P}` is defined as a L2 nodal projection of the
pressure gradient. Note that the notion of a provisional velocity,
:math:`\hat u_i`, is used to signify that this velocity is the product
of the momentum solve. The difference between the projected nodal
gradient interpolated to the gauss point and the local gauss point
pressure gradient provides a fourth order pressure stabilization term.
This term can also be viewed as an algebraic form for the momentum
residual. For the continuity equation only, a series of element-based
options that shift the integration points to the edges of the iterated
element is an option.

The Role of :math:`\dot m`
++++++++++++++++++++++++++

In all of the above equations, the advection term is written in terms of
a linearized mass flow rate including a sum over all subcontrol surface
integration points, Eq :eq:`adv-form`. The mass flow rate includes the full
set of stabilization terms obtained from the continuity solve,

.. math::

   \dot m = \left(\bar{\rho} \hat{u}_i + \tau G_i \bar{P} 
     -\tau \frac{\partial \bar{P}}{\partial x_i}\right) n_i\, dS.

The inclusion of the pressure stabilization terms in the advective
transport for the primitives is a required step for ensuring that the
advection velocity is mass conserving. In practice, the mass flow rate
is stored at each integration point in the mesh (edge midpoints for the
edge-based scheme and subcontrol surfaces for the element-based scheme).
When the mixed CVFEM/EBVC scheme is used, the continuity equation solves
for a subcontrol-surface value of the mass flow rate. These values are
assembled to the edge for use in the EBVC discretization approach.
Therefore, the storage for mass flow rate is higher.
