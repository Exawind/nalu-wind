Supported Equation Set
----------------------

This section provides an overview of the currently supported equation
sets. Equations will be decribed in integral form with assumed Favre
averaging. However, the laminar counterparts are supported in the code
base and are obtain in the user file by ommitting a turbulence model
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
above pressure, :math:`\bar P` is the purturbation about the
thermodynamic pressure, :math:`P^{th}`. In a low speed compressible
flow, i.e., flows confined to a closed domain with energy or mass
addition in which the continuity equation has been modifed to accomodate
accoustics, this pressure is interpreted at the thermodynamic pressure
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
In the case of an ABL simulation with flat terrain, the voulme averaging is done
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
:math:`C_p` is the heat capcity,
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
an operator split approach wherebye the set of ODEs are integrated over
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

Shear Stress Transport (SST) RANS Model Suite
+++++++++++++++++++++++++++++++++++++++++++++

Although Nalu-Wind is primarily expected to be a LES simulation tool, RANS
modeling is supported through the activation of the SST equation set.

It has been observed that standard 1998 :math:`k-\omega` models display
a strong sensitivity to the free stream value of :math:`\omega` (see
Mentor, :cite:`Mentor:2003`). To remedy, this, an
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
the Mentor 2003 model are then

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

.. _eqn_sst_des:

Direct Eddy Simulation (DES) Formulation
++++++++++++++++++++++++++++++++++++++++

The DES technique is also supported in the code base when the SST model
is activated. This model seeks to formally relax the RANS-based approach
and allows for a theoretical basis to allow for transient flows. The
method follows the method of Temporally Filtered NS formulation as
decribed by Tieszen, :cite:`Tieszen:2005`.

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
solids surface. Currently, the pressure is transfered from the fluids
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
the scattering coefficeint, :math:`I(s)` is the intensity along the
direction :math:`s_i`, :math:`T` is the temperature and the scalar flux
is :math:`G`. The black body radiation, :math:`I_b`, is defined by
:math:`\frac{\sigma T^4}{\pi}`. Note that for situations in which the
scattering coefficient is zero, the RTE reduces to a set of liniear,
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

RANS and DES simulations using ..math:`k-\omega` :ref:`SST <eqn_komega_sst>` or
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
   d &= \sqrt{\sum_{j=1,3} \left( \frac{\partial \phi}{\partial x_j} \right)^2 \pm 
   \sqrt{\sum_{j=1,3} \left( \frac{\partial \phi}{\partial x_j} \right)^2 + 2 \phi
