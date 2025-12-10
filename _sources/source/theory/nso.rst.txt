Nonlinear Stabilization Operator (NSO)
--------------------------------------

An alternative to classic Peclet number blending is the usage of a
discontinuity capturing operator (DCO), or in the low Mach context a
nonlinear stabilization operator (NSO). In this method, an artifical
viscosity is defined that is a function of the local residual and scaled
computational gradients. Viable usages for the NSO can be
advection/diffusion problems in addition to the aforementioned RTE VMS
approach.

The formal finite element kernel for a NSO approach is as follows,

.. math::
   :label: nsoFEMForm

   \sum_e \int_\Omega \nu(\mathbf{R}) \frac{\partial w}{\partial x_i} g^{ij}
   \frac{\partial \phi} {\partial x_j} d\Omega,
   

where :math:`\nu(\mathbf{R})` is the artifical viscosity which is a
function of the pde fine-scale residual and :math:`g^{ij}` is the
covariant metric tensor).

For completeness, the covariant and contravarient metric tensor are
given by,

.. math::
   :label: coVariant1

   g^{ij} = \frac{\partial x_i} {\partial \xi_k}\frac{\partial x_j} {\partial \xi_k},
   

and

.. math::
   :label: coVariant2

   g_{ij} = \frac{\partial \xi_k} {\partial x_i} \frac{\partial \xi_k} {\partial x_j},
   

where :math:`\xi = (\xi_1, \xi_2, \xi_3)^T`. The form of
:math:`\nu(\mathbf{R})` currently used is as follows,

.. math::
   :label: nuOne

   \nu = \sqrt{ \frac{\mathbf{R_k} \mathbf{R_k}}
   {\frac {\partial \phi}{\partial x_i} g^{ij} \frac{\partial \phi}{\partial x_j}} }.
   

The classic paper by Shakib ( :cite:`Shakib:1991`)
represents the genesis of this method which was done in the
accoustically compressible context.

A residual for a classic advection/diffusion/source pde is simply the
fine scale residual computed at the gauss point,

.. math::
   :label: nsoResidual1

   \mathbf{\hat R} = \frac{\partial \rho \phi}{\partial t}
   + \frac{\partial}{\partial x_j} (\rho u_j \phi - \mu^{eff}
   \frac{\partial \phi}{\partial x_j}) -S
    

Note that the above equation requires a second derivative whose source
is the diffusion term. The first derivative is generally determined by
using projected nodal gradients. As will be noted in the pressure
stabilization section, the advection term carries the pressure
stabilization terms. However, in the above equation, these terms are not
explicity noted. Therefore, an option is to subtract the fine scale
continuity equation to obtain the final general form of the source term,

.. math::
   :label: nsoResidual2

   \mathbf{R} = \mathbf{\hat R} - \phi (\frac{\partial \rho}{\partial t}
   + \frac{\partial \rho u_j }{\partial x_j}).
    

An alternative to the fine-scale PDE is a form that is found by
differencing the linearized form of the residual from the nonlinear
residual,

.. math::
   :label: nsoResidualAlt

   \mathbf{R} = \frac{\partial \rho u_j \phi }{\partial x_j}
   - (\phi \frac{\partial \rho u_j }{\partial x_j} + \rho u_j
   \frac{\partial \phi}{\partial x_j}).
    

The above resembles a commutation error in the nonlinear advection
term.

In general, the NSO-\ :math:`\nu` is prone to percision issues when the
gradients are very close to zero. As such, the value of :math:`\nu` is
limited to a first-order like value. This parameter is proposed as
follows: if an operator were written as a Galerkin (un-stabilized) plus
a diffusion operator, what is the value of the diffusion coefficient
such that first-order upwind is obtained for the combined operator? This
upwind limited value of :math:`\nu` provides the highest value that this
coefficient can (or should) be. The current form of the limited upwind
:math:`\nu` is as follows,

.. math::
   :label: nsoFVForm1

   \nu^{upw} = C_{upw}(\rho u_i g_{ij} \rho u_j )^{\frac{1}{2}}
   

where :math:`C_{upw}` is generally taked to be  0.1.

Using a piecewise-constant test function suitable for CVFEM and EBVC
schemes (the reader is refered to the VMS RTE section), Eq. :eq:`nsoFEMForm`
can be written as,

.. math::
   :label: nsoFVForm2

   -\sum_e \int_\Gamma \nu(\mathbf{R}) g^{ij} \frac{\partial \phi} {\partial x_j} n_i dS.
   

A fourth order form, which writes the stabilization as the difference
between the Gauss-point gradient and the projected nodal gradient
interpolated to the Gauss-point, is also supported,

.. math::
   :label: nsoFVForm4th

   -\sum_e \int_\Gamma \nu(\mathbf{R}) g^{ij}
   (\frac{\partial \phi} {\partial x_j} - G_j \phi ) n_i dS.
   

NSO Based on Kinetic Energy Residual
++++++++++++++++++++++++++++++++++++

An alternative formulation explored is to share the general kernal form
shown in Equation :eq:`nsoFVForm4th`, however, compute :math:`\nu` based on
a fine-scale kinetic energy residual. In this formulation, the
fine-scale kinetic energy residual is obtained from the fine-scale
momentum residual dotted with velocity. As with the continuity
stabilization approach, the fine-scale momentum residual is provided by
Equation :eq:`fineScaleKe`. Therefore, the fine-scale kinetic energy
is written as,

.. math::
   :label: fineScaleKe

   \mathbf{R}_{ke} = \frac{u_j(\frac{\partial p} {\partial x_j} - G_j p )}{2},
   

while the denominator for :math:`\nu` now includes the gradient in ke,

.. math::
   :label: nuKe

   \nu = \sqrt{ \frac{\mathbf{R}_{ke} \mathbf{R}_{ke}}
   {\frac {\partial ke}{\partial x_i} g^{ij} \frac{\partial ke}{\partial x_j}} }.
   

The kinetic energy is simply given by,

.. math::
   :label: keForm

   ke = \frac{u_k u_k}{2}
   

The kinetic energy form of :math:`\nu` is used for all equation sets
with transformation by usage of a turbulent Schmidt/Prandtl number.

Local or Projected NSO Diffusive Flux Coefficient
+++++++++++++++++++++++++++++++++++++++++++++++++

While the NSO kernel is certainly evaluated at the subcontrol surfaces,
the evaluation of :math:`\nu` can be computed by a multitude of
approaches. For example, the artificial diffusive flux coefficient can
be computed locally (with local residuals and local metric tensors) or
can be projected to the nodes (via an :math:`L_{oo}` or :math:`L_2`
projection) and re-interpolated to the gauss points. The former results
in a sharper local value while the later results in a more filtered-like
value. The code base only supports a local NSO :math:`\nu` calculation.

General Findings
++++++++++++++++

In general, the NSO approach seems to work best when running the
fourth-order option as the second-order implementation still looks more
diffuse. When compared to the standard MUSCL-limited scheme, the NSO is
the preferred choice. More work is underway to improve stabilization
methods. Note that a limited set of NSOs are activated in the code base
with specific interest on scalar transport, e.g, momentum, mixture
fraction and static enthalpy transport. When using the :math:`4^{th}`
order method, the consistent mass matrix approach for the projected
nodal gradients is supported for higher order.

NSO as a Turbulence Model
+++++++++++++++++++++++++

The kinetic energy residual form has been suggested to be used as a turbulence model (Guermond and Larios, 2015). However,
inspection of the above NSO kernel form suggests that the model form is not symmetric. Rather, in the context of
turbulence modeling, is closer to the metric tensor acting on the difference between the rate of strain and antisymmetric
tensor. As such, the theory developed, e.g., for eigenvalue perturbations of the stress tensor (see Jofre and Domino, 2017) can not be applied. In this section,
a new form of the NSO is provided in an effort to be used for an LES closure.

In this proposed NSO formulation, the subgrid stress tensor, :math:`\tau^{sgs}_{ij} = \overline{u_i u_j} - \bar u_i \bar u_j`, 
is given by,

.. math::
   :label: nsoTurbForm

   \tau^{sgs}_{ij} = - 2 \rho \nu g^{ij} (S_{ij} -
   \frac{1}{3}\frac{\partial u_k} {\partial x_k} \delta_{ij}) 
   = - 2 \rho \nu g^{ij} S^*_{ij}.


Interestingly, the units of :math:`\nu` are of an inverse time scale while the product :math:`2 \rho \nu g^{ij}` can be viewed
as an non-isotropic eddy viscosity, :math:`\mu^t_{ij}`.

The first order clipping may be relaxed by defining :math:`\nu` as,

.. math::
   :label: nuTurb

   \nu = \frac{| \mathbf{R}_{ke} |} {||ke||_\infty}.


The above form would be closer to what Guermond uses and would avoid the divide-by-zero noted in regions of uniform flow.
