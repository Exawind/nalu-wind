RTE Stabilization
-----------------

The RTE is solved using the method of discrete ordinates using the
symmetric Thurgood quadrature set. The discrete ordinates method is one
in which discrete directions of the intensity are solved. The quadrature
order, :math:`N`, defines the number of ordinate directions that are
solved in a given iteration. In the case of non-scattering media, this
results is a set of decoupled linear partial differential equations. For
the symmetric Thurgood set, the number of ordinate directions is given
by :math:`8N^2`. Values of N that are required for suitable accuracy
starts at :math:`N=2` with more than adequate resolution at :math:`N=4`.

For each ordinate direction, a weight is provided, :math:`w_k` (not to
be confused with the test function :math:`w`). For each intensity
ordinate direction, :math:`I_k`, integrated quantities such as scalar
flux and radiative heat flux are computed as,

.. math:: G = \sum I_k w_k

and,

.. math:: q_j = \sum I_k s_j w_k.

The stabilization that is used in the RTE equation can be placed in the
class of residual-based stabilization. In this particular
implementation, the scaled residual of the RTE equation is added. This
implementation has its roots in the classic variational multiscale
(VMS).

In the VMS framework, the degree of freedom is decomposed in terms of
its resolved and fine scale, :math:`I+I'`. Without specific definition
of the test function, the weighted residual statement for the RTE within
a VMS framework is given by,

.. math::
   :label: rteVMS

   \int w \left( s_i \frac{\partial}{\partial x_i} \left(I\left(s\right) + I'\left(s\right)\right)
   + (\mu_a + \mu_s) \left(I\left(s\right) + I'\left(s\right)\right) - \frac{\mu_a \sigma T^4}{\pi}
   - \frac{\mu_s}{4\pi}G \right) {\rm d}V = 0.


Grouping resolved and fine scale terms results in an equation takes the
form of a standard Galerkin contribution in addition to the fine
structure statement,

.. math::
   :label: rteVMS1
   
   \int w \left( s_i \frac{\partial}{\partial x_i} I\left(s\right)
   + (\mu_a + \mu_s)I - \frac{\mu_a \sigma T^4}{\pi} -\frac{\mu_s}{4\pi}G \right) {\rm d}V  \\ 
   + \int w \left( s_i \frac{\partial}{\partial x_i} I'\left(s\right)
   + (\mu_a + \mu_s) I' \right) {\rm d}V  = 0.

Note that the isotropic source term has not contributed to the VMS
framework other than through the right hand source term.

In general, gradients in the fine scale quantity are to be avoided.
Therefore, the first term in the second line of Eq. :eq:`rteVMS1` is
integrated by parts to yield the following form (note the boundary term,
:math:`\int_\Gamma` that is shown below is frequently dropped)

.. math::
   :label: rteVMS2
   
   \int w \left( s_i \frac{\partial}{\partial x_i} I\left(s\right)
   + (\mu_a +\mu_s)I - \frac{\mu_a \sigma T^4}{\pi} -\frac{\mu_s}{4\pi}G \right) {\rm d}V  \\ 
   - \int I' s_i \frac{\partial w}{\partial x_i} {\rm d}V 
   + \int_{\Gamma} w s_i I' n_i {\rm d}S + \int w (\mu_a + \mu_s)I' {\rm d}V  = 0.


The following ansatz, which now includes the classic stabilization
parameter, :math:`\tau`, provides closure of the above fine scale
equation,

.. math::
   :label: fineScaleClosure

   I' = -\tau\left(s_i \frac{\partial}{\partial x_i} I\left(s\right) + (\mu_a + \mu_s)I\left(s\right) 
      - \frac{\mu_a \sigma T^4}{\pi} -\frac{\mu_s}{4\pi}G \right) = -\tau R(s)


Substituting Eq. :eq:`fineScaleClosure` into Eq. :eq:`rteVMS2` yields,

.. math::
   :label: rteVMS3
   
   \int w \left( s_i \frac{\partial}{\partial x_i} I\left(s\right)
   + (\mu_a +\mu_s)I - \frac{\mu_a \sigma T^4}{\pi} -\frac{\mu_s}{4\pi}G \right) {\rm d}V  \\ 
   + \int \tau s_i \frac{\partial w}{\partial x_i} R(s) {\rm d}V 
   - \int_\Gamma \tau w R(s) s_i n_i {\rm d}S - \int \tau w (\mu_a + \mu_s)R(s) {\rm d}V  = 0.


In the above equation, the residual of the intensity equation for
ordinate :math:`s` is denoted by :math:`R(s)`. A compact form of the
equation is provided by defining a modified test function,
:math:`\tilde w`, (again note retention of the stabilized boundary term)

.. math::
   :label: rteVMSCompact
   
   \int \tilde w \left( s_i \frac{\partial}{\partial x_i} I\left(s\right)
   + (\mu_a + \mu_s)I - \frac{\mu_a \sigma T^4}{\pi} -\frac{\mu_s}{4\pi}G \right) {\rm d}V \\
   - \beta \int_\Gamma \tau w R(s) s_i n_i {\rm d}S = 0.


where :math:`\tilde w` is simply equal to,

.. math::
   :label: modW

   \tilde w  = w + \tau \left( s_j \frac{\partial w }{\partial x_j} + \alpha (\mu_a + \mu_s)w \right).


When :math:`\alpha = -1`, we have the above VMS derivation; for
:math:`\alpha = 1`, Galerkin Least Squares is realized; finally for
:math:`\alpha = 0`, we have SUPG. For any formulation other than VMS,
the residual contribution at the boundaries of the domain is dropped
(:math:`\beta = 0`).

The full residual-based equation is placed in divergence form,

.. math::
   :label: rteDivForm
   
   \int \tilde w \left( \frac{\partial}{\partial x_i} s_i I\left(s\right)
   + (\mu_a + \mu_s) I\left(s\right) - \frac{\mu_a \sigma T^4}{\pi} -\frac{\mu_s}{4\pi}G \right) {\rm d}V \\
   - \beta \int_\Gamma \tau w R(s) s_i n_i {\rm d}S = 0.


and split into its Galerkin and stabilized contributions,

.. math::
   :label: rteDivFormSub1
 
   \int w \left( \frac{\partial}{\partial x_i} s_i I\left(s\right)
   + (\mu_a + \mu_s )I\left(s\right) - \frac{\mu_a \sigma T^4}{\pi} -\frac{\mu_s}{4\pi}G \right) {\rm d}V \\
   +\int \tau s_j \frac{\partial w }{\partial x_j} R(s) {\rm d}V \\
   +\alpha\int \tau w (\mu_a + \mu_s)R(s) {\rm d}V \\
   - \beta \int_\Gamma \tau w R(s) s_i n_i {\rm d}S = 0.


Note that the first term in the above equation is integrated by parts,

.. math::

   \int w \frac{\partial}{\partial x_i} s_i I\left(s\right)
   {\rm d}V = -\int I\left(s\right) s_i \frac{\partial w}{\partial x_i} {\rm d}V 
   + \int_{\Gamma} w s_i I\left(s\right) n_i {\rm d}S.

Again, the usage of :math:`\Gamma` provides emphasis that the
contribution is a boundary (exposed face) condition. Therefore, the full
VMS-based stabilized RTE equation is as follows,

.. math::
   :label: rteDivFormSub2

   & \int \left( -I\left(s\right) s_i \frac{\partial w}{\partial x_i}
   + (\mu_a + \mu_s) I\left(s\right) 
   - \frac{\mu_a \sigma T^4}{\pi} -\frac{\mu_s}{4\pi}G \right) {\rm d}V  \\
   &+ \int_\Gamma w s_i I\left(s\right) n_i {\rm d}S  \\
   &+\int \tau s_j \frac{\partial w }{\partial x_j} R(s) {\rm d}V \\
   &+\alpha\int \tau w (\mu_a + \mu_s)R(s) {\rm d}V  \\
   &- \beta \int_\Gamma \tau w R(s) s_i n_i {\rm d}S = 0.
 

Definition of the test function
+++++++++++++++++++++++++++++++

Following the work of Martinez, :cite:`Martinez:2005`, the
test function is chosen to be a piecewise-constant value within the
control volume, :math:`w = w_I` and zero outside of this control volume
(Heaviside). A key property of this function, as pointed out by
Martinez, is that its gradient is a distribution of delta functions on
the control volume boundary:

.. math::
   :label: eqn:GradPiecewiseConstant

   \frac{\partial w_I}{\partial x_i} = - {\bf n}_I \delta({\bf x} - {\bf x}_{\Gamma_I})


where :math:`\Gamma_I` is boundary of control volume :math:`I` and
:math:`{\bf n}_I` is the outward normal on that boundary. Substituting
this relationship into the residual equation provides the final form of
vertex-centered finite volume RTE stabilized equation,

.. math::
   :label: rteSUCVForm1
  
   \int I\left(s\right) s_i n_i {\rm d}S + \int \left( (\mu_a + \mu_s ) I\left(s\right) 
   - \frac{\mu_a \sigma T^4}{\pi} -\frac{\mu_s}{4\pi}G \right) {\rm d}V  \\
   + \int_\Gamma s_i I\left(s\right) n_i {\rm d}S \\
   - \int \tau R(s) s_i n_i {\rm d}S +\alpha \int \tau (\mu_a +\mu_s) R(s)
   {\rm d}V -\beta \int_\Gamma \tau R(s) s_i n_i {\rm d}S= 0.


Given this equation, either an edge-based or element-based scheme can be
used. For :math:`\alpha = 0` and :math:`\beta = 0`, it is noted that
classic SUCV is obtained. The second line of Eq. :eq:`rteSUCVForm1`
represents a boundary contribution. This is where the intensity boundary
condition (Eq. :eq:`intBc`) is applied. As noted in the RTE equation
section, when :math:`s_j n_j` is greater than zero, the interpolated
intensity based on the surface nodal values is used. However, when
:math:`s_j n_j` is less than zero, the intensity boundary condition
value is used. Since the original RTE equation was integrated by parts,
a natural surface flux contribution is applied. In alternative
discretization approaches, e.g., the SUPG FEM-based Sierra Thermal
Radiation Module: Syrinx code, the RTE is not integrated by parts.
Therefore, no boundary term exists, and, therefore, a dirichlet bc is
applied. At corner nodes, this approach can lead to non-intuitive
approaches since the corner node might have surface facets that are both
incoming and outgoing. Weak integration of the flux term eliminated this
complexity.

The form of :math:`\tau`
++++++++++++++++++++++++

The value of the stabilization parameter :math:`\tau` can take on a
variety of forms. A classic derivation provides the form of :math:`\tau`
to be broken out into two forms, :math:`\tau_{adv} = \frac{h}{2}` and
:math:`\tau_{diff} = \frac{1}{(\mu_a+\mu_s)}`. An ad-hoc blending is
given by,

.. math::
   :label: blendedTau

   \tau  = \left( \frac{1} {\frac{2}{h}^2 + (\mu_a+\mu_s)^2} \right)^ \frac{1}{2}


Finally, the classic GFEM form of :math:`\tau` is given by use of the
metric tensor for the element mapping is noted,

.. math::
   :label: farzin

   \tau = \beta^* [s_i g_{ij}s_j]^{-\frac{1}{2}},


with :math:`\beta^*` equal to unity for SUCV and
:math:`\frac{2}{15^{\frac{1}{2}}}` for FEM.

Pure Edge-based Upwind Method
+++++++++++++++++++++++++++++

The residual-based stabilization apporach can lead to predicting
negative intensities. This is simply due to the fact that the
stabilization approach (SUPG) is a linear approach. Extensions of this
residual-based stabilization to include a discontinuity capturing
operator (DCO) are underway. This adds a non-linear stabilization
approach that will, hopefully, eliminate negative intensity predictions.

Alternatively, a first order upwind approach has been implemented by
using EBVC discretization. At this point, no higher order upwind
extensions have been implemented. For the upwind implementation, the
equation solved is,

.. math::
   :label: rteSUCVForm2

   \int I\left(s\right) s_i n_i {\rm d}S + \int \left( (\mu_a + \mu_s ) I\left(s\right) 
   - \frac{\mu_a \sigma T^4}{\pi} -\frac{\mu_s}{4\pi}G \right) {\rm d}V  \\
   + \int_\Gamma s_i I\left(s\right) n_i {\rm d}S  = 0.
   

In the above equation, the “advection operator”,
:math:`I\left(s\right) s_i n_i {\rm d}S` is approximated as using the
“upwind” intensity, e.g., if :math:`s_j n_j` is greater than zero, the
left nodal value is used.


Finite Element SUPG Form
++++++++++++++++++++++++

For the FEM, the test function is the standard weighting. Assuming a pure SUPG formulation, i.e.,
:math:`\alpha = \beta = 0` in Equation :eq:`rteDivFormSub2`, thereby reducing the final form to the following:

.. math::
   :label: rteFemSUPG

   \int \left( -I\left(s\right) s_i \frac{\partial w}{\partial x_i} + w[(\mu_a + \mu_s) I\left(s\right) 
   - \frac{\mu_a \sigma T^4}{\pi} -\frac{\mu_s}{4\pi}G ] \right) {\rm d}V  \\
   + \int_\Gamma w s_i I\left(s\right) n_i {\rm d}S \\
   +\int \tau s_j \frac{\partial w }{\partial x_j} R(s) {\rm d}V \\

The weak boundary condition is applied in a similar manner as with the CVFEM and EBVC form, however,
using the appropriate FEM test test function definition. Finally, the form of :math:`\tau` follows the above
CVFEM form.
