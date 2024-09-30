Supported Equation Set
----------------------

This section provides an overview of the currently supported equation
sets. Equations will be described in integral form with assumed Favre
averaging. However, the laminar counterparts are supported in the code
base and are obtain in the user file by omitting a turbulence model
specification.

Conservation of Mass
++++++++++++++++++++

The continuity equation is always solved in the variable density form.

.. math::

   \int \frac{\partial \bar{\rho}} {\partial t}\, dV
   + \int \bar{\rho} \widetilde{u}_i  n_i\, dS = 0

Since Nalu-Wind uses equal-order interpolation (variables are collocated)
stabilization is required. The stabilization choice will be developed in
the pressure stabilization section.

Note that the use of a low speed compressible formulation requires that
the fluid density be computed by an equation of state that uses the
thermodynamic pressure. This thermodynamic pressure can either be
computed based on a global energy/mass balance or allowed to be
spatially varying. By modification of the continuity density time
derivative to include the :math:`\frac{\partial \rho}{\partial p}`
sensitivity, an equation that admits acoustic pressure waves is
realized.

.. _supp_eqn_set_mom_cons:

Conservation of Momentum
++++++++++++++++++++++++

The integral form of the Favre-filtered momentum equations used for turbulent transport are

.. math::
   :label: favmom

   \int \frac{\partial \bar{\rho} \widetilde{u}_i}{\partial t} \, {\rm d}V
   + \int \bar{\rho} \widetilde{u}_i \widetilde{u}_j n_j \, {\rm d}S
   =
   \int \widetilde{\sigma}_{ij} n_j \, {\rm d}S
   -\int \tau^{sgs}_{ij} n_j \, {\rm d}S \\
   + \int \left(\bar{\rho} - \rho_{\circ} \right) g_i \, {\rm d}V
   + \int \mathrm{f}_i \, {\rm d}V,

where the subgrid scale turbulent stress :math:`\tau^{sgs}_{ij}` is defined as

.. math::
   :label: sgsStress

   \tau^{sgs}_{ij} \equiv \bar{\rho} ( \widetilde{u_i u_j} -
     \widetilde{u}_i \widetilde{u}_j ).

The term :math:`\mathrm{f}_i` is a body force used to represent
additional momentum sources such as wind turbine
blades, Coriolis effect, driving forces, etc.
The Cauchy stress is provided by,

.. math::

   \sigma_{ij}  = 2 \mu \widetilde S^*_{ij} - \bar P \delta_{ij}

and the traceless rate-of-strain tensor defined as follows:

.. math::

   \widetilde S^*_{ij} = \widetilde S_{ij} - \frac{1}{3} \delta_{ij} \widetilde S_{kk} \\
   = \widetilde S_{ij} - \frac{1}{3} \frac{\partial \widetilde u_k }{\partial x_k}\delta_{ij}.

In a low Mach flow, as described in the low Mach theory section, the
above pressure, :math:`\bar P` is the perturbation about the
thermodynamic pressure, :math:`P^{th}`. In a low speed compressible
flow, i.e., flows confined to a closed domain with energy or mass
addition in which the continuity equation has been modified to accommodate
acoustics, this pressure is interpreted at the thermodynamic pressure
itself.

For LES, :math:`\tau^{sgs}_{ij}` that appears in Equation :eq:`favmom` and
defined in Equation :eq:`sgsStress` represents the subgrid stress tensor that
must be closed. The deviatoric part of the subgrid stress tensor is defined as

.. math::
   :label: deviatoric-stress-les

   \tau^{sgs}_{ij} = \tau^{sgs}_{ij} - \frac{1}{3} \delta_{ij} \tau^{sgs}_{kk}

where the subgrid turbulent kinetic energy is defined as
:math:`\tau^{sgs}_{kk} = 2 \bar \rho k`. Note that here,
k represents the modeled turbulent kinetic energy as is formally defined as,

.. math::

   \bar \rho k = \frac{1}{2} \bar\rho ( \widetilde{u_k u_k} - \widetilde u_k \widetilde u_k).

Model closures can use, Yoshizawa's approach when k is not transported:

.. math::

   \tau^{sgs}_{kk} = 2 C_I \bar \rho \Delta^2 | \widetilde S | ^2.

Above, :math:`| \widetilde S | = \sqrt {2 \widetilde S_{ij} \widetilde S_{ij}}`.

For low Mach-number flows, a vast majority of the turbulent kinetic
energy is contained at resolved scales. For this reason, the subgrid
turbulent kinetic energy is not directly treated and, rather, is included
in the pressure as an additional normal stress.
The Favre-filtered momentum equations then become

.. math::
   :label: mod-mom-les

   &\int \frac{\partial \bar{\rho} \widetilde{u}_i}{\partial t}
   {\rm d}V + \int \bar{\rho} \widetilde{u}_i \widetilde{u}_j n_j {\rm d}S
   + \int \left( \bar{P} + \frac{2}{3} \bar{\rho} k \right)
   n_i {\rm d}S = \\
   & \int 2 (\mu + \mu_t) \left( \widetilde{S}_{ij} - \frac{1}{3}
   \widetilde{S}_{kk} \delta_{ij} \right) n_j {\rm d}S
   + \int \left(\bar{\rho} - \rho_{\circ} \right) g_i {\rm d}V,

where LES closure models for the subgrid turbulent eddy viscosity
:math:`\mu_t` are either the constant coefficient Smagorinsky, WALE or
the constant coefficient :math:`k_{sgs}` model (see the turbulence
section).

.. _earth_coriolis_force:

Earth Coriolis Force
~~~~~~~~~~~~~~~~~~~~

For simulation of large-scale atmospheric flows, the following Coriolis
force term can be added to the right-hand-side of the momentum equation (:eq:`favmom`):

.. math::
   :label: cor-term

   \mathrm{f}_i = -2\bar{\rho}\epsilon_{ijk}\Omega_ju_k .

Here, :math:`\Omega` is the Earth's angular velocity vector,
and :math:`\epsilon_{ijk}` is the Levi-Civita symbol denoting the cross product
of the Earth's angular velocity with the local fluid velocity
vector. Consider an "East-North-Up" coordinate system on the Earth's
surface, with the domain centered on a latitude angle :math:`\phi` (changes
in latitude within the computational domain are neglected). In this
coordinate system, the integrand of (cor-term), or the Coriolis
acceleration vector, is

.. math::
   :label: coracc

   2 \bar{\rho} \omega
   \begin{bmatrix} u_n \sin\phi - u_u \cos\phi \\
                   -u_e \sin\phi \\
                   u_e \cos\phi
   \end{bmatrix},

where :math:`\omega \equiv ||\Omega||`.  Often, in geophysical flows it is
assumed that the vertical component of velocity is small and that the
vertical component of the acceleration is small relative to gravity,
such that the terms containing :math:`\cos\phi` are neglected.  However,
there is evidence that this so-called traditional approximation is not
valid for some mesoscale atmospheric phenomena \cite{Gerkema_etal:08},
and so the full Coriolis term is retained in Nalu-Wind. The implementation
proceeds by first finding the velocity vector in the East-North-Up
coordinate system, then calculating the Coriolis acceleration vector
(:eq:`coracc`), then transforming this vector back to the model
:math:`x-y-z` coordinate system.  The coordinate transformations are made
using user-supplied North and East unit vectors given in the model
coordinate system.

.. _boussinesq_buoyancy_model:

Boussinesq Buoyancy Model
~~~~~~~~~~~~~~~~~~~~~~~~~

In atmospheric and other flows, the density differences in the domain can be small
enough as to not significantly affect inertia, but nonetheless the buoyancy term,

.. math::
   :label: buoyancy

   \int \left(\bar{\rho} - \rho_{\circ} \right) g_i ~{\rm d}V,

may still be important.  The Boussinesq model ignores the effect of density on inertia
while retaining the buoyancy term in Equation :eq:`favmom`.  For the purpose of evaluating 
the buoyant force, the density is approximated as

.. math::
   :label: boussdensity

   \frac{\rho}{\rho_{\circ}} \approx 1 - \beta (T-T_{\circ}),

This leads to a buoyancy body force term that depends on temperature (:math:`T`), 
a reference density (:math:`\rho_{\circ}`), a reference temperature (:math:`T_{\circ}`),
and a thermal expansion coefficient (:math:`\beta`) as

.. math::
   :label: boussbuoy

   \int -\rho_{\circ} \beta (T-T_{\circ}) g_i ~{\rm d}V.

The flow is otherwise kept as constant density.


Filtered Mixture Fraction
+++++++++++++++++++++++++

The optional quantity used to identify the chemical state is the mixture
fraction, :math:`Z`. While there are many different definitions of the
mixture fraction that have subtle variations that attempt to capture
effects like differential diffusion, they can all be interpreted as a
local mass fraction of the chemical elements that originated in the fuel
stream. The mixture fraction is a conserved scalar that varies between
zero in the secondary stream and unity in the primary stream and is
transported in laminar flow by the equation,

.. math::
   :label: lam_Z

   \frac{\partial \rho Z}{\partial t}
   + \frac{ \partial \rho u_j Z }{ \partial x_j}
   = \frac{\partial}{\partial x_j} \left( \rho D \frac{\partial Z}{\partial x_j}
   \right),

where :math:`D` is an effective molecular mass diffusivity.

Applying either temporal Favre filtering for RANS-based treatments or
spatial Favre filtering for LES-based treatments yields

.. math::
   :label: turb_Z

   \int \frac{\partial \bar{\rho} \widetilde{Z}}{\partial t} {\rm d}V
   + \int \bar{\rho} \widetilde{u}_j \widetilde{Z} n_j {\rm d}S
   = - \int \tau^{sgs}_{Z,j} n_j {\rm d}S + \int \bar{\rho} D
   \frac{\partial \widetilde{Z}}{\partial x_j} n_j {\rm d}S,

where sub-filter correlations have been neglected in the molecular
diffusive flux vector and the turbulent diffusive flux vector is defined
as

.. math::

   \tau^{sgs}_{Z,j} \equiv \bar{\rho} \left( \widetilde{Z u_j} -
   \widetilde{Z} \widetilde{u}_j \right).

This subgrid scale closure is modeled using the gradient diffusion hypothesis,

.. math::

   \tau^{sgs}_{Z,j} = - \bar{\rho} D_t \frac{\partial Z}{\partial x_j},

where :math:`D_t` is the turbulent mass diffusivity, modeled as
:math:`\bar{\rho} D_t = \mu_t / \mathrm{Sc}_t` where :math:`\mu_t` is the modeled turbulent
viscosity from momentum transport and :math:`\mathrm{Sc}_t` is the
turbulent Schmidt number. The molecular mass diffusivity is then
expressed similarly as :math:`\bar{\rho} D = \mu / \mathrm{Sc}` so that
the final modeled form of the filtered mixture fraction transport
equation is

.. math::

   \frac{\partial \bar{\rho} \widetilde{Z}}{\partial t}
     + \frac{ \partial \bar{\rho} \widetilde{u}_j \widetilde{Z} }{ \partial x_j}
     = \frac{\partial}{\partial x_j}
       \left[ \left( \frac{\mu}{\mathrm{Sc}} + \frac{\mu_t}{\mathrm{Sc}_t} \right)
       \frac{\partial \widetilde{Z}}{\partial x_j} \right].

In integral form the mixture fraction transport equation is

.. math::

   \int \frac{\partial \bar{\rho} \widetilde{Z}}{\partial t}\, dV
     + \int \bar{\rho} \widetilde{u}_j \widetilde{Z} n_j\, dS
     = \int \left( \frac{\mu}{\mathrm{Sc}} + \frac{\mu_t}{\mathrm{Sc}_t} \right)
       \frac{\partial \widetilde{Z}}{\partial x_j} n_j\, dS.

Conservation of Energy
++++++++++++++++++++++

The integral form of the Favre-filtered static enthalpy energy equation
used for turbulent transport is

.. math::
   :label: fav-enth

     \int \frac{\partial \bar{\rho} \widetilde{h}}{\partial t} {\rm d}V
     + \int \bar{\rho} \widetilde{h} \widetilde{u}_j n_j {\rm d}S
     &= - \int \bar{q}_j n_j {\rm d}S
     - \int \tau^{sgs}_{h,j} n_j {\rm d}S
     - \int \frac{\partial \bar{q}_i^r}{\partial x_i} {\rm d}V \\
     &+ \int \left( \frac{\partial \bar{P}}{\partial t}
     + \widetilde{u}_j \frac{\partial \bar{P}}{\partial x_j} \right){\rm d}V
     + \int \overline{\tau_{ij} \frac{\partial u_i}{\partial x_j }} {\rm d}V
     + \int S_\theta {\rm d}V.

The above equation is derived by starting with the total internal
energy equation, subtracting the mechanical energy equation and
enforcing the variable density continuity equation. Note that the above
equation includes possible source terms due to thermal radiatitive
transport, viscous dissipation, pressure work,
and external driving sources (:math:`S_\theta`).

The simple Fickian diffusion velocity approximation,
Equation :eq:`diffvel1`, is assumed, so that the mean diffusive heat flux
vector :math:`\bar{q}_j` is

.. math::

     \bar{q}_j = - \overline{ \left[ \frac{\kappa}{C_p} \frac{\partial h}{\partial x_j}
     - \frac{\mu}{\rm Pr} \sum_{k=1}^K h_k \frac{\partial Y_k} {\partial x_j} \right] }
     - \overline{ \frac{\mu}{\rm Sc} \sum_{k=1}^K h_k \frac{\partial Y_k}{\partial x_j} }.

If :math:`Sc = Pr`, i.e., unity Lewis number (:math:`Le = 1`), then the diffusive heat
flux vector simplifies to :math:`\bar{q}_j = -\frac{\mu}{\mathrm{Pr}}
\frac{\partial \widetilde{h}}{\partial x_j}`. In the code base, the user has
the ability to either specify a laminar Prandtl number, which is a
constant, or provide a property evaluator for thermal conductivity.
Inclusion of a Prandtl number prevails and ensures that the thermal
conductivity is computed base on :math:`\kappa = \frac{C_p \mu}{Pr}`.
The viscous dissipation term is closed by

.. math::

   \overline{\tau_{ij} \frac{\partial u_i}{\partial x_j }}
   &= \left(\left(\mu + \mu_t\right) \left( \frac{\partial \widetilde{u}_i}{\partial x_j}
   + \frac{\partial \widetilde{u}_j}{\partial x_i} \right)
   - \frac{2}{3} \left( \bar{\rho} \widetilde{k} +
   \mu_t \frac{\partial \widetilde{u}_k}{\partial x_k} \right)
   \delta_{ij} \right) \frac{\partial \widetilde{u}_i}{\partial x_j}
   \\
   &= \left[ 2 \mu \widetilde{S}_{ij}
   + 2 \mu_t \left( \widetilde{S}_{ij} - \frac{1}{3} \widetilde{S}_{kk}
   \delta_{ij} \right) - \frac{2}{3} \bar{\rho} \widetilde{k}
   \delta_{ij} \right] \frac{\partial \widetilde{u}_i}{\partial x_j}.

The subgrid scale turbulent flux vector :math:`\tau^{sgs}_{h}` in
Equation :eq:`fav-enth` is defined as

.. math::

   \tau_{h u_j} \equiv \bar{\rho} \left( \widetilde{h u_j} -
        \widetilde{h} \widetilde{u}_j \right).

As with species transport, the gradient diffusion hypothesis is used to close
this subgrid scale model,

.. math::

   \tau^{sgs}_{h,j} = - \frac{\mu_t}{\mathrm{Pr}_t} \frac{\partial \widetilde{h}}{\partial x_j},

where :math:`\mathrm{Pr}_t` is the turbulent Prandtl number and :math:`\mu_t` is
the modeled turbulent eddy viscosity from momentum closure.
The resulting filtered and modeled turbulent energy equation is given by,

.. math::
   :label: mod-enth

   \int \frac{\partial \bar{\rho} \widetilde{h}}{\partial t} {\rm d}V
   + \int \bar{\rho} \widetilde{h} \widetilde{u}_j n_j {\rm d}S
   &= \int \left( \frac{\mu}{\rm Pr} + \frac{\mu_t}{{\rm Pr}_t} \right)
   \frac{\partial \widetilde{h}}{\partial x_j}  n_j {\rm d}S
   - \int \frac{\partial \bar{q}_i^r}{\partial x_i} {\rm d}V \\
   &+ \int \left( \frac{\partial \bar{P}}{\partial t} + \widetilde{u}_j
   \frac{\partial \bar{P}}{\partial x_j}\right){\rm d}V
   + \int \overline{\tau_{ij} \frac{\partial u_j}{\partial x_j }} {\rm d}V.


The turbulent Prandtl number must have the same value as the turbulent
Schmidt number for species transport to maintain unity Lewis number.

Review of Prandtl, Schmidt and Unity Lewis Number
+++++++++++++++++++++++++++++++++++++++++++++++++

For situations where a single diffusion coefficient is applicable (e.g.,
a binary gas system) the Lewis number is defined as:

.. math::
   :label: lewisNumber

   {\rm Le} = \frac{\rm Sc}{\rm Pr} = \frac{\alpha}{D}.


If the diffusion rates of energy and mass are equal,

.. math::
   :label: lewisNumberUnity

   {\rm Sc = Pr \ and \ Le = 1}.


For completeness, the thermal diffusivity, Prandtl and Schmidt number
are defined by,

.. math::
   :label: thermalDiff

   \alpha = \frac{\kappa}{\rho c_p},


.. math::
   :label: prandtl

   {\rm Pr} = \frac{c_p \mu }{\kappa} = \frac{\mu}{\rho \alpha},


and

.. math::
   :label: schmidt

   {\rm Sc} = \frac{\mu }{\rho D},


where :math:`c_p` is the specific heat, :math:`\kappa`, is the thermal
conductivity and :math:`\alpha` is the thermal diffusivity.

Thermal Heat Conduction
+++++++++++++++++++++++

For non-isothermal object response that may occur in conjugate heat
transfer applications, a simple single material heat conduction equation
is supported.

.. math::
   :label: thermalHeatEquation

   \int \rho C_p \frac{\partial T} {\partial t} {\rm d}V
   + \int q_j n_j {\rm d}S = \int S {\rm d}V.


where :math:`q_j` is again the energy flux vector, however, now in the
following temperature form:

.. math::

   q_j = -\kappa \frac{\partial T}{\partial x_j}.

.. _abl_forcing_term:

ABL Forcing Source Terms
++++++++++++++++++++++++

In LES of wind plant atmospheric flows, it is often necessary to
drive the flow to a predetermined vertical velocity and/or temperature profile.
In Nalu-Wind, this is achieved by adding appropriate
source terms :math:`\mathrm{f}_i` to the
momentum equation :eq:`favmom` and
:math:`S_\theta` to the enthalpy equation :eq:`fav-enth`.

First, the momentum source term is discussed.
The main objective of this implementation is to force the volume averaged velocity at
a certain location to a specified value (:math:`<\mathrm{u}_i>=\mathrm{U}_i`).
The brackets used here, :math:`<>`, mean volume averaging over a certain region.
In order to achieve this, a source term must be applied to the momentum equation.
This source term can be better understood as a proportional controller within the
momentum equation.

The velocity and density fields can be decomposed into a volume averaged component
and fluctuations about that volume average as
:math:`\mathrm{u}_i = \left< \mathrm{u}_i \right> + \mathrm{u}_i'` and
:math:`\bar{\rho} = \left< \bar{\rho} \right> + \bar{\rho}'`.
A decomposition of the plane averaged momentum at a given instance in time is then

.. math::
       \left< \bar{\rho}  \mathrm{u}_i  \right>  =
        \left< \bar{\rho} \right> \left< \mathrm{u}_i \right>
        + \left< \bar{\rho}'  \mathrm{u}'_i  \right>.

We now wish to apply a momentum source based on a desired spatial averaged velocity
:math:`\mathrm{U}_i`.
This can be expressed as:

.. math::
       \left< \bar{\rho}  \mathrm{u}_i^*  \right>  =
        \left< \bar{\rho} \right> \left< \mathrm{u}^*_i \right>
        + \left< \bar{\rho}'  {\mathrm{u}^*_i}'  \right>,

where :math:`\mathrm{u}_i^*` is an unknown reference velocity field whose volume
average is the desired  velocity :math:`\left< \mathrm{u}_i^* \right> = \mathrm{U}_i`.
Since the correlation :math:`\left< \bar{\rho}'  \mathrm{u^*}'_i  \right>`
is unknown, we assume that

.. math::
    \left< \bar{\rho}'  \mathrm{u^*}'_i  \right>
    =
    \left< \bar{\rho}'  \mathrm{u}'_i  \right>

such that the momentum source can now be defined as:

.. math::
   :label: abl-mom-source

   {\mathrm{f}_i} = \alpha_u
        \left(  \, \frac{\left< \bar{\rho} \right> \mathrm{U_i}
        - \left< \bar{\rho} \right> \left< \mathrm{u}_i \right>}
        {\Delta t}\right)

where :math:`\left< \right>` denotes volume averaging at a
certain time :math:`t`,
:math:`\mathrm{U}_i` is the desired spatial averaged
velocity,
and :math:`\Delta t` is the time-scale between when the source term is computed
(time :math:`t`) and when it is applied (time :math:`t + \Delta t`).
This is typically chosen to be the simulation time-step.
In the case of an ABL simulation with flat terrain, the volume averaging is done
over an infinitesimally thin slice in the :math:`x` and :math:`y` directions,
such that the body force is only a
function of height :math:`z` and time :math:`t`.
The implementation allows the
user to prescribe relaxation factors :math:`\alpha_u` for the source terms that are
applied. Nalu-Wind uses a default value of 1.0 for the relaxation factors if no
values are defined in the input file during initialization.

The enthalpy source term works similarly to the momentum source term.
A temperature difference is computed at every time-step and a forcing term
is added to the enthalpy equation:

.. math::

  S_\theta = \alpha_\theta C_p
      \left(
         \frac{\theta_{\rm ref} - \left< \theta \right>}{\Delta t}
      \right)

where :math:`\theta_{\rm ref}` is the desired spatial averaged temperature,
:math:`\left< \theta \right>` is the spatial averaged temperature,
:math:`C_p` is the heat capacity,
:math:`\alpha_\theta` is the relaxation factor,
and
:math:`\Delta t` is the time-scale.

The present implementation can vary the
source terms as a function of time and space using either a user-defined table
of previously computed source terms (e.g., from a *precursor* simulation or
another model such as WRF), or compute the source term as a function of the
transient flow solution.

Conservation of Species
+++++++++++++++++++++++

The integral form of the Favre-filtered species equation used for
turbulent transport is

.. math::
   :label: fav-species

   \int \frac{\partial \bar{\rho} \widetilde{Y}_k}{\partial t} {\rm d}V
   + \int \bar{\rho} \widetilde{Y}_k \widetilde{u}_j n_j {\rm d}S =
   - \int \tau^{sgs}_{Y_k,j} n_j {\rm d}S
   - \int \overline{\rho Y_k \hat{u}_{j,k}} n_j {\rm d}S +
   \int \overline{\dot{\omega}_k} {\rm d}V,


where the form of diffusion velocities (see Equation :eq:`diffvel1`)
assumes the Fickian approximation with a constant value of diffusion
velocity for consistency with the turbulent form of the energy equation,
Equation :eq:`fav-enth`. The simplest form is Fickian diffusion with the
same value of mass diffusivity for all species,

.. math::
   :label: diffvel1

   \hat{u}_{j,k}= - D \frac{1}{Y_k}
   \frac{\partial Y_k}{\partial x_j} .


The subgrid scale turbulent diffusive flux vector :math:`\tau^{sgs}_{Y_kj}` is defined
as

.. math::

   \tau^{sgs}_{Y_k,j} \equiv \bar{\rho} \left( \widetilde{Y_k u_j} -
   \widetilde{Y_k} \widetilde{u}_j \right).

The closure for this model takes on its usual gradient diffusion hypothesis, i.e.,

.. math::

   \tau^{sgs}_{Y_k,j} = - \frac{\mu_t}{\mathrm{Sc}_t} \frac{\partial
     \widetilde{Y}_k}{\partial x_j},

where :math:`\mathrm{Sc}_t` is the turbulent Schmidt number for all
species and :math:`\mu_t` is the modeled turbulent eddy viscosity from
momentum closure.

The Favre-filtered and modeled turbulent species transport equation is,

.. math::
   :label: mod-species

   \int \frac{\partial \bar{\rho} \widetilde{Y}_k}{\partial t} {\rm d}V
   + \int \bar{\rho} \widetilde{Y}_k \widetilde{u}_j n_j {\rm d}S =
   \int \left( \frac{\mu}{\rm Sc}
   + \frac{\mu_t}{{\rm Sc}_t}  \right)
   \frac{\partial \widetilde{Y}_k}{\partial x_j} n_j {\rm d}S +
   \int \overline{\dot{\omega}}_k {\rm d}V .


If transporting both energy and species equations, the laminar Prandtl
number must be equal to the laminar Schmidt number and the turbulent
Prandtl number must be equal to the turbulent Schmidt number to maintain
unity Lewis number. Although there is a species conservation equation
for each species in a mixture of :math:`n` species, only :math:`n-1`
species equations need to be solved since the mass fractions sum to
unity and

.. math::

   \widetilde{Y}_n = 1 - \sum_{j \ne n}^{n} \widetilde{Y}_j .

Finally, the reaction rate source term is expected to be added based on
an operator split approach whereby the set of ODEs are integrated over
the full time step. The chemical kinetic source terms can be
sub-integrated within a time step using a stiff ODE integrator package.

The following system of ODEs are numerically integrated over a time step
:math:`\Delta t` for a fixed temperature and pressure starting from the
initial values of gas phase mass fraction and density,

.. math::

   \dot{Y}_k = \frac{\dot{\omega}_k \left( Y_k \right) }{\rho} \ .

The sources for the sub-integration are computed with the composition
and density at the new time level which are used to approximate a mean
production rate for the time step

.. math::

   \dot{\omega}_k \approx \frac{\rho^{\ast} Y^{\ast}_k - \rho Y_k}{\Delta t} \ .

.. _theory_ksgs_les_model:

Subgrid-Scale Kinetic Energy One-Equation LES Model
+++++++++++++++++++++++++++++++++++++++++++++++++++

The subgrid scale kinetic energy one-equation turbulence model, or
:math:`k^{sgs}` model, :cite:`Davidson:1997`, represents a
simple LES closure model. The transport equation for subgrid turbulent
kinetic energy is given by

.. math::
   :label: ksgs

   \int \frac{\partial \bar{\rho}{k^\mathrm{sgs}}}{\partial t} {\rm d}V
   + \int \bar{\rho}{k^\mathrm{sgs}} \widetilde{u}_j n_j {\rm d}S =
   \int \frac{\mu_t}{\sigma_k} \frac{\partial {k^\mathrm{sgs}}}{\partial x_j} n_j {\rm d}S +
   \int \left(P_k^\mathrm{sgs} - D_k^\mathrm{sgs}\right) {\rm d}V.


The production of subgrid turbulent kinetic energy, :math:`P_k^\mathrm{sgs}`, is modeled by,

.. math::
   :label: mod-prod

   P_k \equiv -\overline{\rho u_i'' u_j''} \frac{\partial \widetilde{u}_i}{\partial x_j},


while the dissipation of turbulent kinetic energy, :math:`D_k^\mathrm{sgs}`, is given by

.. math::

   D_k^\mathrm{sgs} = \rho C_{\epsilon} \frac{{k^\mathrm{sgs}}^{\frac{3}{2}}}{\Delta},

where the grid filter length, :math:`\Delta`, is given in terms of the
grid cell volume by

.. math:: \Delta = V^{\frac{1}{3}}.

The subgrid turbulent eddy viscosity is then provided by

.. math:: \mu_t = C_{\mu_{\epsilon}} \Delta {k^\mathrm{sgs}}^{\frac{1}{2}},

where the values of :math:`C_{\epsilon}` and :math:`C_{\mu_{\epsilon}}`
are 0.845 and 0.0856, respectively.

For simulations in which a buoyancy source term is desired, the code supports the Rodi form,

.. math:: P_b = \beta \frac{\mu^T}{Pr} g_i \frac{\partial T}{\partial x_i}.

.. _eqn_komega_sst:

RANS Model Suite
++++++++++++++++

Although Nalu-Wind is primarily expected to be a LES simulation tool,
RANS modeling is supported through the activation of different
two-equation RANS models: the Chien :math:`k-\epsilon` model
:cite:`chien1982predictions`, the Wilcox 1998 :math:`k-\omega` model
:cite:`wilcox1998turbulence`, and the SST model. For the first two
models, the reader is referred to the reference papers and the NASA
Turbulence Modeling Resource for the `Chien
<https://turbmodels.larc.nasa.gov/ke-chien.html>`_ and `Wilcox
<https://turbmodels.larc.nasa.gov/wilcox.html>`_ models. The SST model
is explained in more details below.

Shear Stress Transport (SST) Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It has been observed that standard 1998 :math:`k-\omega` models display
a strong sensitivity to the free stream value of :math:`\omega` (see
Menter, :cite:`Menter:2003`). To remedy, this, an
alternative set of transport equations have been used that are based on
smoothly blending the :math:`k-\omega` model near a wall with
:math:`k-\epsilon` away from the wall. Because of the relationship
between :math:`\omega` and :math:`\epsilon`, the transport equations for
turbulent kinetic energy and dissipation can be transformed into
equations involving :math:`k` and :math:`\omega`. Aside from constants,
the transport equation for :math:`k` is unchanged. However, an
additional cross-diffusion term is present in the :math:`\omega`
equation. Blending is introduced by using smoothing which is a function
of the distance from the wall, :math:`F(y)`. The transport equations for
the Menter 2003 model are then

.. math::

   \int \frac{\partial \bar{\rho} k}{\partial t} \text{d}V
   + \int \bar{\rho} k\widetilde{u}_{j} n_{j} \text{d} S =
   \int {(\mu + \hat \sigma_k \mu_{t})} \frac{\partial k}{\partial x_{j}} n_{j}
   + \int \left(P_{k}^{\omega} - \beta^* \bar{\rho} k \omega\right) \text{d} V,

.. math::

   \int \frac{\partial \bar{\rho} \omega}{\partial t}\text{d} V
   + \int \bar{\rho} \omega \widetilde{u}_{j} n_{j} \text{d}S =
   \int  {(\mu + \hat\sigma_{\omega} \mu_{t})} \frac{\partial \omega}{\partial x_{j}} n_{j}
   + \int {2(1-F) \frac{\bar{\rho}\sigma_{\omega2}} {\omega}
   \frac{\partial k}{\partial x_j} \frac{\partial \omega}{\partial x_j} } \text{d}V \\
   + \int \left(\frac{\hat\gamma}{\nu_t} P_{k}^{\omega} -
   \hat \beta \bar{\rho} \omega^{2}\right) \text{d}V.

where the value of :math:`\beta^*` is 0.09.

The model coefficients, :math:`\hat\sigma_k`, :math:`\hat\sigma_{\omega}`, :math:`\hat\gamma` and :math:`\hat\beta`
must also be blended, which is represented by

.. math::

   \hat \phi = F\phi_1+ (1-F)\phi_2.

where :math:`\sigma_{k1} = 0.85`, :math:`\sigma_{k2} = 1.0`,
:math:`\sigma_{\omega1} = 0.5`, :math:`\sigma_{\omega2} = 0.856`,
:math:`\gamma_1 = \frac{5}{9}`, :math:`\gamma_2 = 0.44`,
:math:`\beta_1 = 0.075` and :math:`\beta_2 = 0.0828`. The blending
function is given by

.. math::

   F = \tanh(arg_{1}^{4}),

where

.. math::

   arg_{1} = \min \left( \max \left( \frac{\sqrt{k}}{\beta^* \omega y},
   \frac{500 \mu}{\bar{\rho} y^{2} \omega} \right),
   \frac{4 \bar{\rho} \sigma_{\omega2} k}{CD_{k\omega} y^{2}} \right).

The final parameter is

.. math::

   CD_{k\omega} = \max \left( 2 \bar{\rho} \sigma_{\omega2} \frac{1}{\omega}
   \frac{\partial k}{\partial x_{j}} \frac{\partial \omega}{\partial x_{j}}, 10^{-10} \right).

An important component of the SST model is the different expression used
for the turbulent viscosity,

.. math::

   \mu_{t} = \frac {a_1 \bar{\rho} k} {\max\left( a_1 \omega, S F_2 \right) },

where :math:`F_2` is another blending function given by

.. math::

   F_2 = \tanh(arg_{2}^{2}).

The final parameter is

.. math::

   arg_{2} = \max\left( \frac{2 \sqrt{k}}{\beta^* \omega y},
   \frac{500 \mu}{\bar{\rho} \omega y^{2}} \right).

The Menter SST Two-Equation Model with Controlled Decay (SST-SUST) is
also supported, :cite:`Spalart:2007`. Two new constants are added that
are incorporated into additional source terms for the transport
equations:

.. math::

   + \int \left(\beta^* \bar{\rho} k_{amb} \omega_{amb}\right) \text{d} V,

.. math::

   + \int \left(\hat \beta \bar{\rho} \omega_{amb}^{2}\right) \text{d}V.

where the constants are :math:`k_{amb}` and
:math:`\omega_{amb}`. Typically these are set to :math:`k_{amb} =
10^{-6} U_{\infty}^2` and :math:`\omega_{amb} = \frac{5 U_\infty}{L}`,
where :math:`L` is a defining length scale for the particular problem,
and :math:`U_\infty` is the freestream velocity. The value chosen for
these constants should match the values for :math:`\omega` and
:math:`k` at the inflow BC.

.. _sst_limiter:

SST Mixing Length Limiter
~~~~~~~~~~~~~~~~~~~~~~~~~

When using SST to model the Atmospheric Boundary Layer with the Coriolis effect, a mixing length limiter should be included. The limiter included here is based on the limiter for the k-epsilon model in :cite:`Koblitz:2013`. An analogous limiter was derived for the SST model. The SST limiter was presented in :cite:`Adcock:2021` and will be written up in a future publication.

The mixing length limiter replaces the SST model parameter, :math:`\gamma`, in the :math:`\omega` equation with :math:`\gamma^*`. :math:`\gamma^*` is a blend of :math:`\gamma_1^*` and :math:`\gamma_2^*` using the SST blending function, :math:`F`

.. math::
   \gamma^* = F \gamma_1^* + (1-F) \gamma_2^*.

:math:`\gamma_i^*` for :math:`i=1,2` is calculated from :math:`C_{\varepsilon 1, i}^*` as

.. math::
   \gamma_i^* = C_{\varepsilon 1, i}^* -1.

:math:`C_{\varepsilon 1, i}^*` is calculated by applying the mixing length limiter to :math:`C_{\varepsilon 1, i}` as

.. math::
   C_{\varepsilon 1, i}^* = C_{\varepsilon 1,i} + (C_{\varepsilon 2,i} - C_{\varepsilon 1, i} ) \frac{l_t}{l_e}.

:math:`C_{\varepsilon 1, i}` is calculated from the SST model constant :math:`gamma_1` as

.. math::
   C_{\varepsilon 1, i} = \gamma_i + 1.

:math:`C_{\varepsilon 2, i}` is calculated from the SST model constants :math:`\beta_i` and :math:`\beta^*` as

.. math::
   C_{\varepsilon 1, i} = \frac{\beta_i}{\beta^*} + 1.

The maximum mixing length, :math:`l_e` is calculated as

.. math::
   l_e = .00027 G / f_c,

where :math:`G` is the geostrophic (freestream) velocity and :math:`f_c` is the Coriolis force. The Coriolis force is calculated as

.. math::
   f_c = 2 \Omega sin \lambda,

where :math:`\Omega` is the earth's angular velocity and :math:`\lambda` is the latitude. 

.. _eqn_sst_des:

Laminar-Turbulent Transition Model Formulation
++++++++++++++++++++++++++++++++++++++++++++++
To account for the effects of the laminar-turbulent boundary layer transition, 
Menter's one-equation :math:`\gamma` transition model :cite:`Menter:2015` is coupled with the SST model.
The model consists of single transport equation for intermittency

.. math::
   \frac{D(\rho \gamma)}{Dt}=P_{\gamma}-D_{\gamma}+\frac{\partial }{\partial x_j}\left[ (\mu + \frac{\mu_t}{\sigma_{\gamma}} )\frac{\partial \gamma}{\partial x_j} \right]

The production term, :math:`P_{\gamma}`, and destruction term, :math:`D_{\gamma}`, are as:

.. math::
   P_{\gamma}=F_{length} \rho S \gamma (1-\gamma) F_{onset}, \quad D_{\gamma}=C_{a2} \rho \Omega \gamma F_{turb} (C_{e2}\gamma-1)

The model constants are:

.. math::
   F_{length}=100, \quad  c_{e2}=50, \quad  c_{a2}=0.06, \quad  \sigma_{\gamma}=1.0 

The transition onset criteria of the model are defined as:

.. math::
   F_{onset1}=\frac{Re_{v}}{2.2Re_{\theta c}}, \quad F_{onset2}=(F_{onset1},2.0 ) 

.. math::
   F_{onset3}=\max \left(1- \left (\frac{R_{T}}{3.5}\right)^3,0 \right ), \quad F_{onset}=\max(F_{onset2}-F_{onset3},0)

.. math::
   F_{turb}=e^{-\left ( \frac{R_{T}}{2} \right )^{4}}, \quad R_T=\frac{\rho k}{\mu \omega}, \quad Re_v=\frac{\rho d_{w}^2S}{\mu}

The transition onset occurs once the scaled vorticity Reynolds number, :math:`Re_{v}/2.2`, exceeds the critical momentum thickness Reynolds number, :math:`Re_{\theta c}`, from the empirical correlations.

The output intermittency from the transition model is applied to both the production and destruction terms of the turbulent kinetic energy
transport equation. 

Detached Eddy Simulation (DES) Formulation
++++++++++++++++++++++++++++++++++++++++++

The DES technique is also supported in the code base when the SST model
is activated. This model seeks to formally relax the RANS-based approach
and allows for a theoretical basis to allow for transient flows. The
method follows the method of Temporally Filtered NS formulation as
described by Tieszen, :cite:`Tieszen:2005`.

The SST DES model simply changes the turbulent kinetic energy equation
to include a new minimum scale that manipulates the dissipation term.

.. math::

   D_k = \frac{\rho k^{3/2}} {l_{DES}},

where :math:`l_{DES}` is the min(\ :math:`l_{SST}, c_{DES}l_{DES}`). The
constants are given by, :math:`l_{SST}=\frac{k^{1/2}}{\beta^* \omega}`
and :math:`c_{DES}` represents a blended set of DES constants:
:math:`c_{{DES}_1} = 0.78` and :math:`c_{{DES}_2} = 0.61`. The length
scale, :math:`l_{DES}` is the maximum edge length scale touching a given
node.

Active Model Split (AMS) Formulation
++++++++++++++++++++++++++++++++++++++++++++

The AMS approach is a recently developed hybrid RANS/LES framework by Haering, 
Oliver and Moser :cite:`Haering-etal:2020`.  In this approach a triple
decomposition is used, breaking the instantaneous velocity field into
an average component :math:`<u_i>`, a resolved fluctuation :math:`u_i^>` 
and an unresolved fluctuation :math:`u_i^<`.  The subgrid stress is
split into two terms, :math:`\tau_{ij} = \tau_{ij}^{SGRS} + 
\tau_{ij}^{SGET}`, with :math:`\tau_{ij}^{SGRS}` modeling the mean 
subgrid stress, taking on the form of a standard RANS subgrid stress
model and :math:`\tau_{ij}^{SGET}` representing the energy transfer
from the resolved to the modeled scales.  In addition, a forcing stress
:math:`F_i` is added to the momentum equations to induce the transfer
of energy from the modeled to the resolved scales.  Thus the AMS
approach solves the following momentum equation

.. math::
   :label: ams-mom-les

   &\int \frac{\partial \bar{\rho} \widetilde{u}_i}{\partial t}
   {\rm d}V + \int \bar{\rho} \widetilde{u}_i \widetilde{u}_j n_j {\rm d}S
   + \int \left( \bar{P} + \frac{2}{3} \bar{\rho} k \right)
   n_i {\rm d}S = \\
   & \int 2 \mu \left( \widetilde{S}_{ij} - \frac{1}{3}
   \widetilde{S}_{kk} \delta_{ij} \right) n_j {\rm d}S
   + \int \left(\bar{\rho} - \rho_{\circ} \right) g_i {\rm d}V + \\
   & \int 2 \mu_t \left( <S_{ij}> - \frac{1}{3}
   <S_{kk}> \delta_{ij} \right) n_j {\rm d}S
   + \int \left( \mu_{ik}^t \widetilde{u}_j + \mu_{jk}^t \widetilde{u}_i \right) n_k {\rm d}S 
   + \int f_i {\rm d}V. 

   
Split subgrid model stress
~~~~~~~~~~~~~~~~~~~~~~~~~~

In a typical Hybrid RANS/LES approach, the observation that the LES
and RANS equations take on the same mathematical form is leveraged,
relying simply on a modified turbulent viscosity coefficient that
takes into account the ability to resolve some turbulent content.  Due
to deficiencies in this approach as discussed in Haering et
al. :cite:`Haering-etal:2020`, an alternative approach where the modeled term
is split into two contributions, one representing the impact on the
mean flow and one the impact on the resolved fluctuations, from the
unresolved content, is used in the Active Model Split (AMS)
formulation.

Starting by substitution of a triple decomposition of the flow
variables, :math:`\phi = \langle \phi \rangle + \phi^> + \phi^<`, with
:math:`\langle \cdot \rangle` representing a mean quantity, :math:`\phi^>` a
resolved fluctuation and :math:`\phi^<` an unresolved fluctuation and
dropping all terms that have an unresolved fluctuation in them (since
by definition, these terms cannot be resolved and thus must be
modeled) we get the following momentum equation:

.. math::

   \frac{\partial \overline{u_i}}{\partial t} +
   \frac{\partial \overline{u_i} \ \overline{u_j}}{\partial x_j}
   = -\frac{1}{\rho} \left( \frac{\partial \overline{P}}{\partial x_i} \right) + 
   \nu \frac{\partial^2 \overline{u_i}}{\partial x_j \partial x_j} + 
   \frac{\partial \tau_{ij}^M}{\partial x_j} + F_i


Note that here, :math:`\overline{\phi} = \langle \phi \rangle + \phi^>`
represents an instantaneous resolved quantity and :math:`F_i` is a forcing
term discussed in Sec. :ref:`AMS forcing <amsforcing>`.

The model term in AMS, :math:`\tau_{ij}^M` is split into two pieces,
the first representing the impact of the unresolved scales on the mean
flow, referred to as :math:`\tau_{ij}^{SGRS}`, since this mimics the
purpose of RANS models and seeks to model the subgrid Reynolds Stress
(SGRS).  The second term represents the impact of the unresolved
scales on the resolved fluctuations which acts to capture energy
transfer from the resolved fluctuations to the unresolved
fluctuations, which as Haering et al. points out, is really the
primary function of typical LES SGS models.  As this term models
subgrid energy transfer (SGET), it is referred to as
:math:`\tau_{ij}^{SGET}`.

:math:`\tau_{ij}^{SGRS}` is modeled using a typical RANS model, but since in
the hybrid context, some turbulence is resolved, the magnitude of the
stress tensor is reduced through a derived scaling with :math:`\alpha = \beta^{1.7}`, 
:math:`\beta = 1 - k_{res}/k_{tot}`, where :math:`k_{tot}` is the total 
kinetic energy, taken from the RANS model and 
:math:`k_{res} = 0.5 \langle u_i^> u_i^> \rangle`, a
measure of the average resolved turbulent kinetic energy.

:math:`\tau_{ij}^{SGET}` is modeled using the M43 SGS model discussed
in Haering et al. :cite:`HaeringAIAA`.  This uses an anisotropic
representation, :math:`\tau_{ij} = \nu_{jk} \partial_k u_i + \nu_{ik}
\partial_k u_j`, of the stress tensor and a tensorial eddy viscosity,
:math:`\nu_{ij} = C(\mathcal{M}) \langle \epsilon \rangle^{1/3}
\mathcal{M}_{ij}^{4/3}`, with :math:`C`, a coefficient determined as a
function of the eigenvalues of :math:`\mathcal{M}`, a metric tensor
measure of the grid and :math:`\langle \epsilon \rangle` inferred from
the RANS model.

So this produces the final form for the AMS model term,

.. math::

   \begin{aligned}
   \tau_{ij}^M &= \tau_{ij}^{SGRS} + \tau_{ij}^{SGET} \\
   &= 2 \alpha (2 - \alpha) \nu^{RANS}_t \langle S_{ij} \rangle - \frac{2}{3} \beta k_{tot} \delta_{ij} + C(\mathcal{M}) \langle \epsilon \rangle^{1/3} \left( \mathcal{M}_{jk}^{4/3} \frac{\partial u_i^>}{\partial x_k} + \mathcal{M}_{ik}^{4/3} \frac{\partial u_j^>}{\partial x_k} \right).
   \end{aligned}

The AMS model terms are implemented for the edge based
scheme in *MomentumSSTAMSDiffEdgeKernel*.
The isotropic component, :math:`2 \beta k_{tot}\delta_{ij}/3` is
brought into the pressure term.

Averaging functions
~~~~~~~~~~~~~~~~~~~

The averaging algorithms are invoked as part of the
*AMSAlgDriver* and are called from the *pre_iter_work*
function so that they are executed at the beginning of each Picard 
iteration. The *AMSAlgDriver* is invoked last, so to ensure
that this is also done initially, so that the initial step has correct
average quantities, the averaging functions are also called in the
*initial_work* function.

The main averaging algorithm is
*SSTAMSAveragesAlg*. The averaging function is
solving a simple causal average equation for the intermediate (or final) quantity:

.. math::

   \frac{\partial \langle \phi^{*} \rangle}{\partial t} = \frac{1}{T_{RANS}^{*}}\left( \phi^{*} - \langle \phi^{n} \rangle \right)

Here :math:`\langle \cdot \rangle` refers to a mean (time-averaged)
quantity and :math:`T_{RANS}` is the timescale of the turbulence
determined by the underlying RANS scalars (:math:`1 / (\beta^*\omega)`
in SST).  :math:`()^{n}` refers to a previous timestep quantity and
:math:`()^{*}` refers to an intermediate quantity. 
Note that currently the time scale is stored in a nodal
field.

We can discretize the causal average equation explicitly to produce
the implemented form:

.. math::

   \begin{aligned}
   \langle \phi^{*} \rangle &= \langle \phi^{n} \rangle + \frac{\Delta t}{T_{RANS}^{*}}\left( \phi^{*} - \langle \phi^{n} \rangle \right) \\
   \langle \phi^{*} \rangle &= \frac{\Delta t}{T_{RANS}^{*}}\phi^{*} + \left( 1 - \frac{\Delta t}{T_{RANS}^{*}} \right) \langle \phi^{n} \rangle
   \end{aligned}

The averaging is carried out for velocities (:math:`u_i`), velocity
gradients (:math:`\partial u_i / \partial x_j`), pressure (:math:`P`),
density (:math:`\rho`), resolved turbulent kinetic energy
(:math:`k_{res} = 0.5 u^>_i u^>_i`) and the kinetic energy production
:math:`\left( \mathcal{P}_k = \langle S_{ij} \rangle \left(
\tau_{ij}^{SGRS} - u^>_i u^>_j \right) \right)`.  Note that :math:`^>`
is used to denote a resolved fluctuation, i.e. :math:`\phi^> = \phi -
\langle \phi \rangle`.

Dynamic Hybrid Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~

Typically in a hybrid model, it is necessary to have diagnostics that
assess the ability of the grid to resolve turbulent content and to aid
in its introduction.  In AMS, there are two main diagnostic
quantities, :math:`\alpha = \beta^{1.7} = (1 - k_{res}/k_{tot})^{1.7}` and a resolution
adequacy parameter, :math:`r_k`, which is used to evaluate where in
the flow the grid and the amount of resolved turbulent content is
inconsistent.

:math:`\beta` is a straight-forward calculation.  Limiters are
applied to :math:`\beta` to realize the RANS and DNS limits.  In the
RANS limit, :math:`k_{res} = 0` and thus :math:`\beta = 1`, so
:math:`\beta` is limited from evaluation above 1.  In the DNS limit,
ideally, the ratio of the approximate Kolmogorov velocity scale to total TKE
would be used as a lower bound,

.. math::

   \beta = \max \left( 1 - \frac{k_{res}}{k_{tot}}, \frac{(\nu \epsilon)^{1/2}}{k_{tot}} \right),

but that has shown some performance issues near the wall when using SST with AMS.  
Currently an adhoc lower bound of :math:`\beta = 0.01` is used instead.  
The resolution adequacy parameter is based on the ratio between the
anisotropic grid metric tensor, :math:`\mathcal{M} = \mathcal{J}^T J`, where
:math:`\mathcal{J}` is the mapping from a unit cube to the cell, and the
length scale associated with the model production.  It takes the form,

.. math::

   r_k = \left( \frac{3}{2 \langle \overline{v}^2 \rangle} \right)^{3/2} \max_{\lambda}(\mathcal{P}_{ik}^{SGS} \mathcal{M}_{kj}).

For the RANS models used in Nalu-Wind, :math:`\langle \overline{v}^2 \rangle
\approx 5\nu_{RANS}/T_{RANS}`.  :math:`\mathcal{P}_{ij}^{SGS} =
\frac{1}{2} ( \tau_{ik} \partial \overline{u}_k / \partial x_j +
\tau_{jk} \partial \overline{u}_k / \partial x_i)` is the full subgrid
production tensor, with :math:`\tau_{ij} = \tau_{ij}^{SGRS} +
\tau_{ij}^{SGET} + \frac{2}{3} \alpha k_{tot} \delta_{ij}`.


.. _amsforcing:

Forcing Term
~~~~~~~~~~~~

When the grid is capable of resolving some turbulent content, the
model will want to reduce the modeled stress and allow for resolved
turbulence to contribute the remaining piece of the total stress.  As
discussed in Haering et al. :cite:`Haering-etal:2020` and the observation of
"modeled-stress depletion" in other hybrid approaches, such as DES, a
mechanism for inducing resolved turbulent fluctuations at proper
energy levels and timescales to match your reduction of the modeled
stress is needed.  AMS resolves this issue through the use of an
active forcing term, designed to introduce turbulent fluctuations into
regions of the grid where turbulent content can be supported.  The
implications of the specific form and method of introduction for this
forcing term is still an area of ongoing research, but for now,
empirical testing has shown great success with a simple turbulent
structure based off of Taylor-Green vortices.

The forcing term :math:`F_i` is determined by first specifying an
auxiliary field based off of a Taylor-Green vortex:

.. math::

   \begin{aligned}
   h_1 &= \frac{1}{3} \cos(a_x x'_1) \sin(a_y x'_2) \sin(a_z x'_3), \\
   h_2 &= - \sin(a_x x'_1) \cos(a_y x'_2) \sin(a_z x'_3), \\
   h_3 &= \frac{2}{3} \sin(a_x x'_1) \sin(a_y x'_2) \cos(a_z x'_3), \\
   \end{aligned}

with :math:`\mathbf{x'} = \mathbf{x} + \langle \mathbf{u} \rangle t`
and :math:`a_i = \pi / \mathbb{P}_i`. :math:`\mathbb{P}` is determined
as follows,

.. math::

   \begin{aligned}
   l  &= \frac{4 - (1 - \max(\beta, 0.8))}{0.4}\frac{(\beta k)^{3/2}}{\epsilon} \\
   l  &= \min \left( \max \left( l, 70 \frac{\nu^{3/4}}{\epsilon^{1/4}} \right), d \right) \\\\
   l'_i &= \min \left( l, L_{p_i} \right) \\
   f_i  &= \mathrm{nint}\left( \frac{L_{p_i}}{l'_i} \right) \\
   \mathbb{P}_i &= \frac{L_{p_i}}{f_i},
   \end{aligned}

where :math:`L_{p_i}` is related to the periodic domain size and is
user specified.  With the initial TG vortex field, :math:`h_i`,
determined, we now determine a scaling factor (:math:`\eta`) for the
forcing.

.. math::

   \begin{aligned}
   T_\beta &= \max \left( \beta k / \epsilon, 6 \sqrt{\nu / \epsilon} \right) \\
   F_{tar} &= 8 \sqrt{\alpha \overline{v}^2} / T_\beta \\\\
   \mathcal{P}_r &= \Delta t F_{tar} \left( h_i u_i^{>} \right) \\\\
   \beta_{K} &= \min(\sqrt{\nu \epsilon / k}, 1) \\\\
   \hat{\beta} &= \left \{ 
   \begin{aligned}
   \frac{1 - \beta}{1 - \beta_{K}} &\qquad \beta_{K} < 1 \\
   10000 &\qquad \mathrm{else}
   \end{aligned}
   \right. \\\\
   C_f &= -1  \tanh(1 - \frac{1}{\sqrt{\min(\langle r_k \rangle}, 1)}) \\\\
   C_f &= C_f  (1.0 - \min(\tanh(10  (\hat{\beta} - 1)) + 1, 1)) \\\\
   \eta &= \left \{ 
   \begin{aligned}
   F_{tar} C_f &\qquad \langle r_k \rangle < 1, \ \mathcal{P}_r \ge 0 \\
   0 &\qquad \mathrm{else}
   \end{aligned}
   \right. \\
   \end{aligned}

Now the final forcing field, :math:`F_i = \eta h_i`.  Since this is
being added as a source term to the momentum solve, we
are not projecting onto a divergence free field and are instead
allowing that to pass into the continuity solve, where the
intermediate velocity field with the forcing will then be projected
onto a divergence free field. This is implemented in the node kernels as
*MomentumSSTAMSForcingNodeKernel*.

AMS with SST Mixing Length Limiter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using AMS with SST as the mean (RANS) contribution to model the Atmospheric Boundary Layer with the Coriolis effect, SST should include a mixing length limiter. The mixing length limiter is described in :ref:`SST Mixing Length Limiter <sst_limiter>`. For consistency, when using the limiter the RANS time scale, :math:`T_{RANS}^*`, should depend on the mixing length rather than :math:`\omega` to account for the effect of the limiter. The time scale becomes

.. math::
   T_{RANS}^* = \frac{l_t}{\sqrt{k}}.

Solid Stress
++++++++++++

A fully implicit CVFEM (only) linear elastic equation is supported in
the code base. This equation is either used for true solid stress
prediction or for smoothing the mesh due to boundary mesh motion (either
through fluid structure interaction (FSI) or prescribed mesh motion).

Consider the displacement for component i, :math:`u_i` equation set,

.. math::
   :label: linearElastic

   \rho \frac{\partial^2 u_i} {{\partial t}^2}
   - \frac{\partial \sigma_{ij}}{\partial x_j} = F_i,


where the Cauchy stress tensor, :math:`\sigma_{ij}` assuming Hooke’s law
is given by,

.. math::
   :label: stress

   \sigma_{ij} = \mu \left ( \frac{\partial u_i}{\partial x_j}
   + \frac{\partial u_j}{\partial x_i} \right)
   + \lambda \frac{\partial u_k}{\partial x_k} \delta_{ij}.


Above, the so-called Lame coefficients, Lame’s first parameter,
:math:`\lambda` (also known as the Lame modulus) and Lame’s second
parameter, :math:`\mu` (also known as the shear modulus) are provided as
functions of the Young’s modulus, :math:`E`, and Poisson’s ratio,
:math:`\nu`; here shown in the context of a isotropic elastic material,

.. math::
   :label: lame_mu

   \mu = \frac{E}{2\left(1+\nu\right)},


and

.. math::
   :label: lame_lambda

   \lambda = \frac{E \nu}{\left(1+\nu\right) \left(1-2 \nu \right)}.


Note that the above notation of :math:`u_i` to represent displacement is
with respect to the classic definition of current and model coordinates,

.. math::
   :label: displacement2

   x_i = X_i + u_i


where :math:`x_i` is the position, relative to the fixed, or previous
position, :math:`X_i`.

The above equations are solved for mesh displacements, :math:`u_i`. The
supplemental relationship for solid velocity, :math:`v_i` is given by,

.. math::
   :label: velocity

   v_i = \frac{\partial u_i}{\partial t}.


Numerically, the velocity might be obtained by a backward Euler or BDF2
scheme,

.. math::
   :label: mesh_velocity

   v_i = \frac{\gamma_1 u^{n+1}_i + \gamma_2 u^n_i + \gamma_3 u^{n-1}_i}{\Delta t}


Moving Mesh
+++++++++++

The code base supports three notions of moving mesh: 1) linear elastic
equation system that computes the stress of a solid 2) solid body
rotation mesh motion and 3) mesh deformation via an external
source.

The linear elastic equation system is activated via the standard
equation system approach. Properties for the solid are specified in the
material block. Mesh motion is prescribed by the input file via the
``mesh_motion`` block. Here, it is assumed
that the mesh motion is solid rotation. For fluid/structure interaction
(FSI) a mesh smoothing scheme is used to propagate the surface mesh
displacement obtained by the solids solve. Simple mesh smoothing is
obtained via a linear elastic solve in which the so-called Lame
constants are proportional to the inverse of the dual volume. This
allows for boundary layer mesh locations to be stiff while free stream
mesh elements to be soft.

Additional mesh motion terms are required for the Eulerian fluid
mechanics solve. Using the geometric conservative law the time and
advection source term for a general scalar :math:`\phi` can be written
as:

.. math::
   :label: gcl

   \int \frac {\rho \phi } {\partial t}\, dV
   + \int \rho \phi \left ( u_j - v_j \right) n_j\, dS
   + \int \rho \phi \frac{\partial v_k}{\partial x_j}\, dV,


where :math:`u_j` is the fluid velocity and :math:`v_j` is the mesh
velocity. Mesh velocities and the mesh velocity spatial derivatives are
provided by the mesh smoothing solve. Activating the external mesh
deformation or mesh motion block will result in the velocity relative to
mesh calculation in the advection terms. The line command for source
term, ":math:`gcl`" must be activated for each equation for the volume
integral to be included in the set of PDE solves. Finally, transfers are
expected between the physics. For example, the solids solve is to
provide mesh displacements to the mesh smoothing realm. The mesh
smoothing procedure provides the boundary velocity, mesh velocity and
projected nodal gradients of the mesh velocity to the fluids realm.
Finally, the fluids solve is to provide the surface force at the desired
solids surface. Currently, the pressure is transferred from the fluids
realm to the solids realm. The ideal view of FSI is to solve the solids
pde at the half time step. As such, in time, the
:math:`P^{n+\frac{1}{2}}` is expected. The
``fsi_interface`` input line command attribute is
expected to be set at these unique surfaces. More advanced FSI coupling
techniques are under development by a current academic partner.

Radiative Transport Equation
++++++++++++++++++++++++++++

The spatial variation of the radiative intensity corresponding to a
given direction and at a given wavelength within a radiatively
participating material, :math:`I(s)`, is governed by the Boltzmann
transport equation. In general, the Boltzmann equation represents a
balance between absorption, emission, out-scattering, and in-scattering
of radiation at a point. For combustion applications, however, the
steady form of the Boltzmann equation is appropriate since the transient
term only becomes important on nanosecond time scales which is orders of
magnitude shorter than the fastest chemical.

Experimental data shows that the radiative properties for heavily
sooting, fuel-rich hydrocarbon diffusion flames (:math:`10^{-4}`\ % to
:math:`10^{-6}`\ % soot by volume) are dominated by the soot phase and
to a lesser extent by the gas phase. Since soot emits and absorbs
radiation in a relatively constant spectrum, it is common to ignore
wavelength effects when modeling radiative transport in these
environments. Additionally, scattering from soot particles commonly
generated by hydrocarbon flames is several orders of magnitude smaller
that the absorption effect and may be neglected. Moreover, the phase
function is rarely known. However, for situations in which the phase
function can be approximated by the iso-tropic scattering assumption,
i.e., an intensity for direction :math:`k` has equal probability to be
scattered in any direction :math:`l`, the appropriate form of the
Botzmann radiative transport is

.. math::
   :label: lam-scalar-flux

   s_i \frac{\partial}{\partial x_i} I\left(s\right)
   + \left(\mu_a + \mu_s \right) I\left(s\right) =
   \frac{\mu_a \sigma T^4}{\pi} + \frac{\mu_s}{4\pi}G,


where :math:`\mu_a` is the absorption coeffiecient, :math:`\mu_s` is
the scattering coefficient, :math:`I(s)` is the intensity along the
direction :math:`s_i`, :math:`T` is the temperature and the scalar flux
is :math:`G`. The black body radiation, :math:`I_b`, is defined by
:math:`\frac{\sigma T^4}{\pi}`. Note that for situations in which the
scattering coefficient is zero, the RTE reduces to a set of linear,
decoupled equations for each intensity to be solved.

The flux divergence may be written as a difference between the radiative
emission and mean incident radiation at a point,

.. math::
   :label: div-qrad

   \frac{\partial q_i^r}{\partial x_i} =
       \mu_a \left[ 4 \sigma T^4 - G \right] ,


where :math:`G` is again the scalar flux. The flux divergence term is
the same regardless of whether or not scattering is active. The
quantity, :math:`G/4\pi`, is often referred to as the mean incident
intensity. Note that when the scattering coefficient is non-zero, the
RTE is coupled over all intensity directions by the scalar flux
relationship.

The scalar flux and radiative flux vector represent angular moments of
the directional radiative intensity at a point,

.. math::

   G = \int_{0}^{2\pi}\!\int_{0}^{\pi}\! I\left(s\right)
           \sin \theta_{zn} d \theta_{zn} d \theta_{az} ,

.. math::

   q^{r}_{i} = \int_{0}^{2\pi}\!\int_{0}^{\pi}\! I\left(s\right)
           s_i \sin \theta_{zn} d \theta_{zn} d \theta_{az} ,

where :math:`\theta_{zn}` and :math:`\theta_{az}` are the zenith and
azimuthal angles respectively as shown in Figure :numref:`ord-dir`.

.. _ord-dir:

.. figure:: images/ordinate.pdf
   :alt: Ordinate Direction Definition
   :width: 500px
   :align: center

   Ordinate Direction Definition,
   :math:`{\bf s} = \sin \theta_{zn} \sin \theta_{az} {\bf i} + \cos \theta_{zn} {\bf j} + \sin \theta_{zn} \cos \theta_{az} {\bf k}`.

The radiation intensity must be defined at all portions of the boundary
along which :math:`s_i n_i < 0`, where :math:`n_i` is the outward
directed unit normal vector at the surface. The intensity is applied as
a weak flux boundary condition which is determined from the surface
properties and temperature. The diffuse surface assumption provides
reasonable accuracy for many engineering combustion applications. The
intensity leaving a diffuse surface in all directions is given by

.. math::
   :label: intBc2

   I\left(s\right) = \frac{1}{\pi} \left[ \tau \sigma T_\infty^4
                   + \epsilon \sigma T_w^4
                   + \left(1 - \epsilon - \tau \right) K \right] ,

where :math:`\epsilon` is the total normal emissivity of the surface,
:math:`\tau` is the transmissivity of the surface, :math:`T_w` is the
temperature of the boundary, :math:`T_\infty` is the environmental
temperature and :math:`H` is the incident radiation, or irradiation
(incoming radiative flux). Recall that the relationship given by
Kirchoff’s Law that relates emissivity, transmissivity and reflectivity,
:math:`\rho`, is

.. math::

   \rho + \tau + \epsilon = 1.

where it is implied that :math:`\alpha = \epsilon`.

Wall Distance Computation
+++++++++++++++++++++++++

RANS and DES simulations using :math:`k-\omega` :ref:`SST <eqn_komega_sst>` or
:ref:`SST-DES <eqn_sst_des>` equations requires the specification of a *wall
distance* for computing various turbulence parameters. For static mesh
simulations this field can be generated using a pre-processing step and provided
as an input in the mesh database. However, for moving mesh simulations, e.g.,
blade resolved wind turbine simulations, this field must be computed throughout
the course of the simulation. Nalu-Wind implements a Poisson equation
(:cite:`Tucker2003`) to determine the wall distance :math:`d` using the
gradients of a field :math:`\phi`.

.. math::

   \nabla^2 \phi &= 1 \\
   d &= \pm\sqrt{\sum_{j=1,3} \left( \frac{\partial \phi}{\partial x_j} \right)^2} + \sqrt{\sum_{j=1,3} \left( \frac{\partial \phi}{\partial x_j} \right)^2 + 2 \phi}
