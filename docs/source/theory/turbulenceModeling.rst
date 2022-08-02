Turbulence Modeling
-------------------

Unlike a RANS approach which models most or all of the turbulent
fluctuations, LES directly solves for all resolved turbulent length
scales and only models the smallest scales below the grid size. In this
way, a majority of the problem-dependent, energy-containing turbulent
structure is directly solved in a model-free fashion. The subgrid scales
are closer to being isotropic than the resolved scales, and they
generally act to dissipate turbulent kinetic energy cascaded down from
the larger scales in momentum-driven turbulent flows. Modeling of these
small scales is generally more straightforward than RANS approaches, and
overall solutions are usually more tolerant to LES modeling errors
because the subgrid scales comprise such a small portion of the overall
turbulent structure. While LES is generally accepted to be much more
accurate than RANS approaches for complex turbulent flows, it is also
significantly more expensive than equivalent RANS simulations due to the
finer grid resolution required. Additionally, since LES results in a
full unsteady solution, the simulation must be run for a long time to
gather any desired time-averaged statistics. The tradeoff between
accuracy and cost must be weighed before choosing one method over the
other.

The separation of turbulent length scales required for LES is obtained
by using a spatial filter rather than the RANS temporal filter. This
filter has the mathematical form

.. math::
   :label: les-filter

   \overline{\phi(\boldsymbol{x},t)} \equiv \int_{-\infty}^{+\infty}
       \phi(\boldsymbol{x}',t) G(\boldsymbol{x}' - \boldsymbol{x})\,
       \mathrm{d}\boldsymbol{x}',


which is a convolution integral over physical space
:math:`\boldsymbol{x}` with the spatially-varying filter function
:math:`G`. The filter function has the normalization property
:math:`\int_{-\infty}^{+\infty}
G(\boldsymbol{x})\, \mathrm{d}\boldsymbol{x} = 1`, and it has a
characteristic length scale :math:`\Delta` so that it filters out
turbulent length scales smaller than this size. In the present
formulation, a simple “box filter” is used for the filter function,

.. math::

   G(\boldsymbol{x}' - \boldsymbol{x}) = \left\{ \begin{array}{l@{\quad:\quad}l}
       1/V         & (\boldsymbol{x}' - \boldsymbol{x}) \in \mathcal{V} \\
       0           & \mathrm{otherwise} \\
       \end{array} \right.,

where :math:`V` is the volume of control volume :math:`\mathcal{V}`
whose central node is located at :math:`\boldsymbol{x}`. This is
essentially an unweighted average over the control volume. The length
scale of this filter is approximated by :math:`\Delta = V^\frac{1}{3}`.
This is typically called the grid filter, as it filters out scales
smaller than the computational grid size.

Similar to the RANS temporal filter, a variable can be represented in
terms of its filtered and subgrid fluctuating components as

.. math:: \phi = \bar{\phi} + \phi'.

For most forms of the filter function :math:`G(\boldsymbol{x})`,
repeated applications of the grid filter to a variable do not yield the
same result. In other words, :math:`\bar{\bar{\phi}} \ne 
\bar{\phi}` and therefore :math:`\overline{\phi'} \ne 0`, unlike with
the RANS temporal averages.

As with the RANS formulation, modeling is much simplified in the
presence of large density variations if a Favre-filtered approach is
used. A Favre-filtered variable :math:`\tilde{\phi}` is defined as

.. math:: \tilde{\phi} \equiv \frac{ \overline{\rho\phi} }{ \bar{\rho} }

and a variable can be decomposed in terms of its Favre-filtered and
subgrid fluctuating component as

.. math:: \phi = \tilde{\phi} + \phi''.

Again, note that the useful identities for the Favre-filtered RANS
variables do not apply, so that
:math:`\bar{\tilde{\phi}} \ne \tilde{\phi}` and
:math:`\overline{\phi''} \ne 0`. The Favre-filtered approach is used for
all LES models in Nalu-Wind.

.. _theory_standard_smagorinsky_les:

Standard Smagorinsky LES Model
++++++++++++++++++++++++++++++

The standard Smagorinsky LES closure model approximates the subgrid
turbulent eddy viscosity using a mixing length-type model, where the LES
grid filter size :math:`\Delta` provides a natural length scale. The
subgrid eddy viscosity is modeled simply as (Smagorinsky)

.. math::
   :label: mut-smag

   \mu_t = \rho \left(C_s \Delta \right)^2 | \tilde {S} |,


The constant coefficient :math:`C_s` typically varies between 0.1 and
0.24 and should be carefully tuned to match the problem being solved
(Rogallo and Moin, :cite:`Rogallo:1984`).  The default value of 0.17 is assigned in the code base.

Although this model is desirable due to its simplicity and efficiency,
care should be taken in its application.  It is known to predict subgrid
turbulent eddy viscosity proportional to the shear rate in the flow,
independent of the local turbulence intensity.  Non-zero subgrid turbulent
eddy viscosity is even predicted in completely laminar regions of the
flow, sometimes even preventing a natural transition to turbulence. The model also
does not asymptotically replicate near wall behavior without either dampening or a
dynamic procedure.

Wall Adapting Local Eddy-Viscosity, WALE
++++++++++++++++++++++++++++++++++++++++

The WALE model of Ducros el al., :cite:`Ducros:1998`,
properly captures the asymptotic behavior for flows that are wall
bounded. In this model, the turbulent viscosity is given by,

.. math::
   :label: mut-wale

   \mu_t = \rho \left(C_w \Delta \right)^2 \frac{\left( S^d_{ij}S^d_{ij}\right)^{3/2}}
   {\left( S_{ij}S_{ij}\right)^{5/2} + \left( S^d_{ij}S^d_{ij}\right)^{5/4}},


with the constant :math:`C_w` of 0.325 and a standard filter,
:math:`\Delta` related to the volume, :math:`V^{\frac{1}{3}}`. The rate
of strain tensor is defined as,

.. math::
   :label: wale-sij

   S_{ij} = \frac{1}{2} \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right)


while :math:`S^d_{ij}` is,

.. math::
   :label: wale-sdij

   S^d_{ij} = \frac{1}{2} \left( g^2_{ij} + g^2_{ji}\right) - \frac{1}{3} \delta_{ij} g^2_{kk}.


Finally, the velocity gradient squared ters are

.. math::
   :label: wale-sqij

    g^2_{ij} = \frac{\partial u_i}{\partial x_k} \frac{\partial u_k}{\partial x_j}


and

.. math::
   :label: wale-gsqji

    g^2_{ji} = \frac{\partial u_j}{\partial x_k} \frac{\partial u_k}{\partial x_i}.


One Equation :math:`k^{sgs}`
++++++++++++++++++++++++++++

See :math:`k^{sgs}` :ref:`PDE <theory_ksgs_les_model>` section.

RANS Models
+++++++++++

As noted, Nalu-Wind supports several different RANS-based model (the
reader is referred to the RANS model equation set description).

Hybrid RANS/LES Models
++++++++++++++++++++++

Nalu-Wind supports the Active Model Split (AMS) hybrid RANS/LES turbulence model :cite:`Haering-etal:2020`. 
The reader is referred to the AMS equation set for more details.

Wall Models
+++++++++++

Flows are either expected to be fully resolved or, alternatively, under-resolved
where wall functions are used. A classic law of the wall has been implemented in
Nalu-Wind. Wall models to handle adverse pressure gradients are planned. For more
information of the form of wall models, please refer to the :ref:`boundary
condition <theory_boundary_conditions>` section of this manual.
