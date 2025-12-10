.. _theory_advection_stabilization:

Advection Stabilization
-----------------------

In general, advection for both the edge and element-based scheme is
identical with standard exception of the location of the integration
points. The full advection term is simply written as,

.. math::
   :label: adv-form

   ADV_{\phi} = \int \rho u_j \phi_{ip} A_j = \sum \dot{m} \phi_{ip},


where :math:`\phi` is :math:`u_i`, :math:`Z`, :math:`h`, etc.

The evaluation of :math:`\phi_{ip}` defines the advection stabilization
choice. In general, the advection choice is a cell Peclet blending
between higher order upwind (:math:`\phi_{upw}`) and a generalized
un-stabilized central (Galerkin) operator, :math:`\phi_{gcds}`,

.. math:: 
   :label: adv-phi-ip

    \phi_{ip} = \eta \phi_{upw} + (1-\eta)\phi_{gcds}.

In the above equation, :math:`\eta` is a cell Peclet
blending. The generalized central operator can take on a pure second
order or pseudo fourth order form (see below). For the classic Peclet
number functional form (see Equation :eq:`classic-pf`) a hybrid upwind
factor, :math:`\gamma`, can be used to ensure that no stabilization is
added (:math:`\eta = 0`) or that full upwind stabilization is included
(as will be shown, even with limiter functions). The hybrid upwind
factor allows one to modify the functional blending function; values of
unity result in the normal blending function response in
Figure :numref:`pec-blend`; values of zero yield a pure central operator, i.e.,
blending function of zero; values :math:`>>` unity result in a blending
function value of unity, i.e., pure upwind. The constant :math:`A` is
implemented with a value of 5. The value of this constant can not be
changed via the input file. Note that this functional form is named the
“classic” form within the input file.

The classic cell Peclet blending function is given by

.. math::
   :label: classic-pf

   \eta = \frac {{\gamma \rm Pe}^2} {5 + {\gamma \rm Pe}^2}.


The classic Peclet functional form makes it difficult to dial in the
exact point at which the Peclet factor transitions from pure upwind to
pure central. Therefore, an alternative form is provided that has a
hyperbolic tangeant functional form. This form allows one to specify the
transition point and the width of the transition (see
Equation :eq:`tanhPF`). The general tanh form is as follows,

.. math::
   :label: tanhPF

   \eta = \frac {1}{2}[(a+b) + (b-a)tanh(\frac{\rm Pe - c_{trans}}{c_{width}})]


Above, the constant :math:`c_{trans}` represents the transition Peclet
number while :math:`c_{width}` represents the width of the transition.
The value of :math:`\lambda` is simply the shift between of the raw tanh
function from zero while :math:`\delta` is the difference between the
max Peclet factor (unity) and the minimum value prior to normalization.
This approach ensures that the function starts at zero and asymptotes to
unity,

.. math::

   \eta = \frac {1}{2}[1+tanh(\frac{\rm Pe - c_{trans}}{c_{width}})].


The cell-Peclet number is computed for each sub-face in the element from
the two adjacent left (L) and right (R) nodes,

.. math::

   {\rm Pe} = \frac{\frac{1}{2} \left( u_{R,i} + u_{L,i} \right) 
              \left( x_{R,i} - x_{L,i} \right) } {\nu }.

A dot-product is implied by repeated indices.

.. _pec-blend:

.. figure:: images/pecletFactor.pdf
   :alt: Cell-Peclet number blending function
   :width: 500px
   :align: center

   Cell-Peclet number blending function outlining classic (varying the
   hybrid factor :math:`\gamma` from 1.0, 0.1 and 0.01; again
   :math:`A=5`) and tanh functional form (:math:`c_{trans}=2000` and
   :math:`c_{width}=200`).

The upwind operator, :math:`\phi_{upw}` is computed based on a blending
of the extrapolated state (using the projected nodal gradient) and the
linear interpolated state. Second or third order upwind is provided
based on the value of :math:`\alpha_{upw}` blending

.. math::
   :label: phi-upwind-full

   \phi_{upw} = \alpha_{upw}\tilde \phi^L_{upw} + \left(1-\alpha_{upw}\right)\phi_{cds}; \dot m > 0, \\
                \alpha_{upw}\tilde\phi^R_{upw} + \left(1-\alpha_{upw}\right)\phi_{cds}; \dot m < 0.


The extrapolated value based on the upwinded left (:math:`\phi^L`) or
right (:math:`\phi^R`) state,

.. math::
   :label: adv-upw-lr

   \tilde \phi^L_{upw} &= \phi^L + d^L_j \frac{\partial \phi^L }{\partial x_j}, \\
   \tilde \phi^R_{upw} &= \phi^R - d^R_j \frac{\partial \phi^R }{\partial x_j}.


The distance vectors are defined based on the distances between the L/R
points and the integration point (for both edge or element-based),

.. math::
   :label: distance-vec

   d^L_j &= x^{ip}_j - x^L_j, \\
   d^R_j &= x^R_j - x^{ip}_j. 

In the case of all transported quantities, a Van Leer
limiter of the extrapolated value is supported and can be activated
within the input file (using the solution options “limiter”
specification).

Second order central is simply written as a linear combination of the
nodal values,

.. math::
   :label: phi-central

   \phi_{cds} = \sum N^{ip}_k \phi_k.


where :math:`N^{ip}_k` is either evaluated at the subcontrol surface or
edge midpoint. In the case of the edge-based scheme, the edge midpoint
evaluation provides for a skew symmetric form of the operator.

The generalized central difference operator is provided by blending with
the extrapolated values and second order Galerkin,

.. math::
   :label: phi4th

   \phi_{gcds} = \frac{1}{2} \left(  \hat\phi^L_{upw} + \hat\phi^R_{upw} \right),


where,

.. math::
   :label: adv-new4th
   
   \hat\phi^L_{upw} &= \alpha \tilde \phi^L_{upw} + \left(1-\alpha\right) \phi_{cds}, \\
   \hat\phi^R_{upw} &= \alpha \tilde \phi^R_{upw} + \left(1-\alpha\right) \phi_{cds}.


The value of :math:`\alpha` provides the type of psuedo fourth order
stencil and is specified in the user input file.

The above set of advection operators can be used to define an idealized
one dimensional stencil denoted by (:math:`i-2`, :math:`i-1`, :math:`i`,
:math:`i+1`, :math:`i+2`), where :math:`i` represents the particular row
for the given transported variable. Below, in the table, the
stencil can be noted for each value of :math:`\alpha` and
:math:`\alpha_{upw}`.

=====================  =====================  ====================  =====================  =====================  ===================  ====================

=====================  =====================  ====================  =====================  =====================  ===================  ====================
:math:`i-2`            :math:`i-1`            :math:`i`             :math:`i+1`            :math:`i+2`            :math:`\alpha`       :math:`\alpha_{upw}`
:math:`0`              :math:`-\frac{1}{2}`   :math:`0`             :math:`+\frac{1}{2}`   :math:`0`              :math:`0`            n/a
:math:`+\frac{1}{8}`   :math:`-\frac{6}{8}`   :math:`0`             :math:`+\frac{6}{8}`   :math:`-\frac{1}{8}`   :math:`\frac{1}{2}`  n/a
:math:`+\frac{1}{12}`  :math:`-\frac{8}{12}`  :math:`0`             :math:`+\frac{8}{12}`  :math:`-\frac{1}{12}`  :math:`\frac{2}{3}`  n/a
:math:`+\frac{1}{4}`   :math:`-\frac{5}{4}`   :math:`+\frac{3}{4}`  :math:`+\frac{1}{4}`   :math:`0`              :math:`\dot m > 0`   :math:`1`
:math:`0`              :math:`-\frac{1}{4}`   :math:`-\frac{3}{4}`  :math:`+\frac{5}{4}`   :math:`-\frac{1}{4}`   :math:`\dot m < 0`   :math:`1`
:math:`+\frac{1}{6}`   :math:`-\frac{6}{6}`   :math:`+\frac{3}{6}`  :math:`+\frac{2}{6}`   :math:`0`              :math:`\dot m > 0`   :math:`\frac{1}{2}`
:math:`0`              :math:`-\frac{2}{6}`   :math:`-\frac{3}{6}`  :math:`+\frac{6}{6}`   :math:`-\frac{1}{6}`   :math:`\dot m < 0`   :math:`\frac{1}{2}`
=====================  =====================  ====================  =====================  =====================  ===================  ====================

It is noted that by varying these numerical parameters, both high
quality, low dissipation operators suitable for LES usage or limited,
monotonic operators suitable for RANS modeling can be accomodated.
