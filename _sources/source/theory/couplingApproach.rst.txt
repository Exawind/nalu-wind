Coupling Approach
-----------------

The classic low Mach implementation uses an incremental approximate
pressure projection scheme in which nonlinear convergence is obtained
using outer Picard loops. Recently a full study on coupling approaches
has been conducted using ASC Algorithm funds. In this project, coupling
methods ranging from fully implicit, fully coupled equal order
pressure/velocity interpolation with pressure stabilization to explicit
advection/diffusion pressure projection schemes. A brief summary of the
results follows.

Specifically, five algorithms were considered and are as follows:

1. A monolithic scheme in which advection and diffusion are implicit using full analytical sensitivities, 
2. Monolithic momentum solve with implicit advection/diffusion in the context of a fourth order stabilized incremental pressure projection scheme, 
3. Monolithic momentum solve with explicit advection; implicit diffusion in the context of a fourth order stabilized incremental pressure projection scheme, 
4. Segregated momentum solve with implicit advection/diffusion in the context of a fourth order stabilized incremental pressure porjectin scheme, and 
5. Explicit momementum advection/diffusion predictor/corrector scheme in the context of a second order stabilized pressure-free approximate projection scheme.


Each of the above algorithms has been run in the context of a transient
uniform flow low Mach flow. The emphasis of this project is transient
flows. As such, the numbers below are to be cast in this context. If
steady flows are desired, conclusions may be different. The slowdown of
each implementation is relative to the core low Mach algorith, i.e.,
algorithm (4) above. Numbers less than unity represent a speed-up
whereas numbers greater than unity represent a slow down: 1) 3.4x, 2)
1.2x, 3) 0.6x, 4) 1.0x, 5) 0.7x.

The above runs were made using a time step that corresponded to a CFL of
slightly less than unity. In this particlar flow, a transitionally
turbulent open jet, the diffusion time scale stability limit was not a
factor. In other words, there existed no detailed boundary layer at the
wall bounded flow at the ground plane. Results for a Reynolds number of 
:math:`45000` back step also are similar to the above jet results.

In general, although a mixture of implicit diffusion and explicit
advection seem to be the winning combination, this scheme is very
sensitive to time step and must be used by an educated user. In general,
the conclusions are, thus far, that the standard segregated pressure
projection scheme is preferred.

The algorithm implemented in Nalu-Wind is a fourth order approximate
projection scheme with monolithic momentum coupling. Evaluation of a
predictor/corrector approach for reating flow is anticipated in the late
FY15 time frame.

.. _theory_errors_splitting_stabilization:

Errors due to Splitting and Stabilization
+++++++++++++++++++++++++++++++++++++++++

As noted in many of our papers, the error in the above method can be
written in block form (let’s relax the variable density nuance - or
simple fold these extra terms into our operators). Here we specifically
partition error into both splitting (the pressure projection aspect of
the alg that factorizes the fully coupled system) and pressure
stabilization. Note that when we run fully coupled simulation with the
same pressure stabilization algorithm, the answers converge to the same
result.

Below, also forgive the specific definitions of :math:`\tau`. In
general, they represent a choice of projection and stabilization time
scales. Finally, the Laplace operator, e.g., :math:`{\bf L_2}`, have the
:math:`\tau`'s built into them.

.. math::
   :label: smoothSplitError

   \left[
       \begin{array}{lr}
         {\bf A}  &  {\bf G}  \\
         {\bf D}  &  {\bf 0}
       \end{array}
     \right]
   %
     \left[
       \begin{array}{l}
         {\bf u}^{n+1}  \\
         p^{n+1} 
       \end{array}
     \right] =
   %
     \left[
       \begin{array}{l}
         {\bf f}  \\
         0
       \end{array}
     \right]    + 
      \left[
       \begin{array}{l}
         ({\bf I}- \tau {\bf A } ){\bf G}(p^{n+1}-p^n) \\ 
         \epsilon({\bf L_i},\tau_i, {\bf D}, {\bf G})
     \end{array}
     \right] 


where the error term that appears for the discrete continuity solve is
given by,

.. math::
   :label: contErrorDefined

   \epsilon({\bf L_i},\tau_i,{\bf D},{\bf G}) =
    (({\bf L_1}-{\bf D}\tau_3{\bf G}) \\
   -({\bf L_2}-{\bf D}\tau_2{\bf G}))(p^{n+1}-p^{n}) \\
   + ({\bf L_2}-{\bf D}\tau_2{\bf G})p^{n+1}


For the sake of this write-up, let :math:`{\bf L_1} = {\bf L_2}` and
:math:`\tau_2 = \tau_3`.
