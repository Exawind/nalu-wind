.. _user_nalu_input_file:

Nalu-Wind Input File
--------------------

Nalu-Wind requires the user to provide an input file, in YAML format, during
invocation at the command line using the :option:`naluX -i` flag. By default,
:program:`naluX` will look for :file:`nalu.i` in the current working directory
to determine the mesh file as well as the run setup for execution. A sample
:download:`nalu.i` is shown below:

.. literalinclude:: nalu.i
   :language: yaml
   :caption: Sample Nalu-Wind input file for the Heat Conduction problem
   :emphasize-lines: 6, 11, 21, 91

Nalu-Wind input file contains the following top-level sections that describe the
simulation to be executed.

**Realms**

  Realms describe the computational domain (via mesh input files) and the set of
  physics equations (Low-Mach Navier-Stokes, Heat Conduction, etc.) that are
  solved over this particular domain. The list can contain multiple
  computational domains (*realms*) that use different meshes as well as solve
  different sets of physics equations and interact via *solution transfer*. This
  section also contains information regarding the initial and boundary
  conditions, solution output and restart options, the linear solvers used to
  solve the linear system of equations, and solution options that govern the
  discretization of the equation set.

  A special case of a realm instance is the input-output realm; this realm type
  does not solve any physics equations, but instead serves one of the following
  purposes:

    - provide time-varying boundary conditions to one or more boundaries within
      one or more of the participating realms in the simulations. In this
      context, it acts as an *input* realm.

    - extract a subset of data for output at a different frequency from the
      other realms. In this context, it acts as an *output* realm.

  Inclusion of an input/output realm will require the user to provide the
  additional :inpfile:`transfers` section in the Nalu-Wind input file that defines
  the solution fields that are transferred between the realms. See
  :ref:`nalu_inp_realm` for detailed documentation on all Realm options.

**Linear Solvers**

  This section configures the solvers and preconditioners used to solve the
  resulting linear system of equations within Nalu-Wind. The linear system
  convergence tolerance and other controls are set here and can be used with
  multiple systems across different realms. See :ref:`nalu_inp_linear_solvers`
  for more details.

**Time Integrators**

  This section configures the time integration scheme used (first/second order
  in time), the duration of simulation, fixed or adaptive timestepping based on
  Courant number constraints, etc. Each time integration section in this list
  can accept one or more :inpfile:`realms` that are integrated in time using
  that specific time integration scheme. See :ref:`nalu_inp_time_integrators`
  for complete documentation of all time integration options available in Nalu-Wind.

**Transfers**

  An optional section that defines one or more solution transfer definitions
  between the participating :inpfile:`realms` during the simulation. Each
  transfer definition provides a mapping of the to and from realm, part, and the
  solution field that must be transferred at every timestep during the
  simulation. See :ref:`nalu_inp_transfers` section for complete documentation of
  all transfer options available in Nalu-Wind.

**Simulations**

  Simulations provides the top-level architecture that orchestrates the
  time-stepping across all the realms and the required equation sets.

.. _nalu_inp_linear_solvers:

Linear Solvers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``linear_solvers`` section contains a list of one or more linear solver
settings that specify the solver, preconditioner, convergence tolerance for
solving a linear system. Every entry in the YAML list will contain the following
entries:

.. note::

   The variable in the :inpfile:`linear_solvers` subsection are prefixed with
   ``linear_solvers.name`` but only the variable name after the period should
   appear in the input file.

.. inpfile:: linear_solvers.name

   The key used to refer to the linear solver configuration in
   :inpfile:`equation_systems.solver_system_specification` section.

.. inpfile:: linear_solvers.type

   The type of solver library used.

   ================== ==========================================================
   Type               Description
   ================== ==========================================================
   ``tpetra``         Tpetra data structures and Belos solvers/preconditioners
   ``hypre``          Hypre data structures and Hypre solver/preconditioners
   ================== ==========================================================

.. inpfile:: linear_solvers.method

   The solver used for solving the linear system.

   When :inpfile:`linear_solvers.type` is ``tpetra`` the valid options are:
   ``gmres``, ``biCgStab``, ``cg``. For ``hypre`` the valid
   options are ``hypre_boomerAMG`` and ``hypre_gmres``.

**Options Common to both Solver Libraries**

.. inpfile:: linear_solvers.preconditioner

   The type of preconditioner used.

   When :inpfile:`linear_solvers.type` is ``tpetra`` the valid options are
   ``sgs``, ``mt_sgs``, ``muelu``. For ``hypre`` the valid
   options are ``boomerAMG`` or ``none``.

.. inpfile:: linear_solvers.tolerance

   The relative tolerance used to determine convergence of the linear system.

.. inpfile:: linear_solvers.max_iterations

   Maximum number of linear solver iterations performed.

.. inpfile:: linear_solvers.kspace

   The Krylov vector space.

.. inpfile:: linear_solvers.output_level

   Verbosity of output from the linear solver during execution.

.. inpfile:: linear_solvers.write_matrix_files

   A boolean flag indicating whether the matrix, the right hand side, and the
   solution vector are written to files during execution. The matrix files are
   written in MatrixMarket format. The default value is ``no``.

**Additional parameters for Belos Solver/Preconditioners**

.. inpfile:: linear_solvers.muelu_xml_file_name

   Only used when the :inpfile:`linear_solvers.preconditioner` is set to
   ``muelu`` and specifies the path to the XML filename that contains various
   configuration parameters for Trilinos MueLu package.

.. inpfile:: linear_solvers.recompute_preconditioner

   A boolean flag indicating whether preconditioner is recomputed during runs.
   The default value is ``yes``.

.. inpfile:: linear_solvers.reuse_preconditioner

   Boolean flag. Default value is ``no``.

.. inpfile:: linear_solvers.summarize_muelu_timer

   Boolean flag indicating whether MueLu timer summary is printed. Default value
   is ``no``.

**Additional parameters for Hypre Solver/Preconditioners**

The user is referred to `Hypre Reference Manual
<https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods/software>`_
for full details on the usage of the parameters described briefly below.

The parameters that start with ``bamg_`` prefix refer to options related to
Hypre's BoomerAMG preconditioner.

.. inpfile:: linear_solvers.bamg_output_level

   The level of verbosity of BoomerAMG preconditioner. See
   ``HYPRE_BoomerAMGSetPrintLevel``. Default: 0.

.. inpfile:: linear_solvers.bamg_coarsen_type

   See ``HYPRE_BoomerAMGSetCoarsenType``. Default: 6

.. inpfile:: linear_solvers.bamg_cycle_type

   See ``HYPRE_BoomerAMGSetCycleType``. Default: 1

.. inpfile:: linear_solvers.bamg_relax_type

   See ``HYPRE_BoomerAMGSetRelaxType``. Default: 6

.. inpfile:: linear_solvers.bamg_relax_order

   See ``HYPRE_BoomerAMGSetRelaxOrder``. Default: 1

.. inpfile:: linear_solvers.bamg_num_sweeps

   See ``HYPRE_BoomerAMGSetNumSweeps``. Default: 2

.. inpfile:: linear_solvers.bamg_max_levels

   See ``HYPRE_BoomerAMGSetMaxLevels``. Default: 20

.. inpfile:: linear_solvers.bamg_strong_threshold

   See ``HYPRE_BoomerAMGSetStrongThreshold``. Default: 0.25

.. _nalu_inp_time_integrators:

Time Integration Options
~~~~~~~~~~~~~~~~~~~~~~~~

.. inpfile:: Time_Integrators

   A list of time-integration options used to advance the :inpfile:`realms` in
   time. Each list entry must contain a YAML mapping with the key indicating the
   type of time integrator. Currently only one option,
   ``StandardTimeIntegrator`` is available.

   .. code-block:: yaml

      Time_Integrators:
        - StandardTimeIntegrator:
            name: ti_1
            start_time: 0.0
            termination_step_count: 10
            time_step: 0.5
            time_stepping_type: fixed
            time_step_count: 0
            second_order_accuracy: yes

            realms:
              - fluids_realm

.. inpfile:: time_int.name

   The lookup key for this time integration entry. This name must match the one
   provided in :inpfile:`Simulations` section.

.. inpfile:: time_int.termination_time

   Nalu-Wind will stop the simulation once the ``termination_time`` has reached.

.. inpfile:: time_int.termination_step_count

   Nalu-Wind will stop the simulation once the specified ``termination_step_count``
   timesteps have been completed. If both :inpfile:`time_int.termination_time`
   and this parameter are provided then this parameter will prevail.

.. inpfile:: time_int.time_step

   The time step (:math:`\Delta t`) used for the simulation. If
   :inpfile:`time_int.time_stepping_type` is ``fixed`` this value does not
   change during the simulation.

.. inpfile:: time_int.start_time

   The starting time step (default: ``0.0``) when starting a new simulation.
   Note that this has no effect on restart which is controlled by
   :inpfile:`restart.restart_time` in the :inpfile:`restart` section.

.. inpfile:: time_int.time_step_count

   The starting timestep counter for a new simulation. See :inpfile:`restart`
   for restarting from a previous simulation.

.. inpfile:: time_int.second_order_accuracy

   A boolean flag indicating whether second-order time integration scheme is
   activated. Default: ``no``.

.. inpfile:: time_int.time_stepping_type

   One of ``fixed`` or ``adaptive`` indicating whether a fixed time-stepping
   scheme or an adaptive timestepping scheme is used for simulations. See
   :inpfile:`time_step_control` for more information on max Courant number based
   adaptive time stepping.

.. inpfile:: time_int.realms

   A list of :inpfile:`realms` names. The names entered here must match
   :inpfile:`name` used in the :inpfile:`realms` section. Names listed here not
   found in :inpfile:`realms` list will trigger an error, while realms not
   included in this list but present in :inpfile:`realms` will not be
   initialized and silently ignored. This can cause the code to abort if the
   user attempts to access the specific realm in the :inpfile:`transfers`
   section.

.. _nalu_inp_realm:

Physics Realm Options
~~~~~~~~~~~~~~~~~~~~~~~

As mentioned previously, :inpfile:`realms` is a YAML list data structure
containing at least one :ref:`nalu_inp_realm` entry that defines the
computational domain (provided as an Exodus-II mesh), the set of physics
equations that must be solved over this domain, along with the necessary initial
and boundary conditions. Each list entry is a YAML dictionary mapping that is
described in this section of the manual. The key subsections of a Realm entry
in the input file are

=============================== =========================================================================
Realm subsection                 Purpose
=============================== =========================================================================
:inpfile:`equation_systems`      Set of physics equations to be solved
:inpfile:`initial_conditions`    Initial conditions for the various fields
:inpfile:`boundary_conditions`   Boundary condition for the different fields
:inpfile:`material_properties`   Material properties (e.g., fluid density, viscosity etc.)
:inpfile:`solution_options`      Discretization and numerical stability
:inpfile:`mesh_motion`           Mesh motion
:inpfile:`output`                Solution output options (file, frequency, etc.)
:inpfile:`restart`               Optional: Restart options (restart time, checkpoint frequency etc.)
:inpfile:`time_step_control`     Optional: Parameters determining variable timestepping
=============================== =========================================================================

In addition to the sections mentioned in the table, there are several additional
sections that could be present depending on the specific simulation type and
post-processing options requested by the user. A brief description of these
optional sections are provided below:

================================ ===========================================================================
Realm subsection                  Purpose
================================ ===========================================================================
:inpfile:`turbulence_averaging`   Generate statistics for the flow field
:inpfile:`post_processing`        Extract integrated data from the simulation
:inpfile:`solution_norm`          Compare the solution error to a reference solution
:inpfile:`data_probes`            Extract data using probes
:inpfile:`actuator`               Model turbine blades/tower using actuator lines
:inpfile:`abl_forcing`            Momentum source term to drive ABL flows to a desired velocity profile
================================ ===========================================================================


Common options
``````````````

.. inpfile:: name

   The name of the realm. The name provided here is used in the
   :inpfile:`Time_Integrators` section to determine the time-integration scheme
   used for this computational domain.

.. inpfile:: mesh

   The name of the Exodus-II mesh file that defines the computational domain for
   this realm. Note that only the base name (i.e., without the ``.NPROCS.IPROC``
   suffix) is provided even for simulations using previously decomposed
   mesh/restart files.

.. inpfile:: automatic_decomposition_type

   Used only for parallel runs, this indicates how the a single mesh database
   must be decomposed amongst the MPI processes during initialization. This
   option should not be used if the mesh has already been decomposed by an
   external utility. Possible values are:

   ==========  ==========================================================
   Value       Description
   ==========  ==========================================================
   rcb         recursive coordinate bisection
   rib         recursive inertial bisection
   linear      elements in order first n/p to proc 0, next to proc 1.
   cyclic      elements handed out to id % proc_count
   ==========  ==========================================================

.. inpfile:: activate_aura

   A boolean flag indicating whether an extra element is *ghosted* across the
   processor boundaries. The default value is ``no``.

.. inpfile:: use_edges

   A boolean flag indicating whether edge based discretization scheme is used
   instead of element based schemes. The default value is ``no``.

.. inpfile:: polynomial_order

   An integer value indicating the polynomial order used for higher-order mesh
   simulations. The default value is ``1``. When :inpfile:`polynomial_order` is
   greater than 1, the Realm has the capability to promote the mesh to
   higher-order during initialization.

.. inpfile:: solve_frequency

   An integer value indicating how often this realm is solved during time
   integration. The default value is ``1``.

.. inpfile:: support_inconsistent_multi_state_restart

   A boolean flag indicating whether restarts are allowed from files where the
   necessary field states are missing. A typical situation is when the
   simulation is restarted using second-order time integration but the restart
   file was created using first-order time integration scheme.

.. inpfile:: activate_memory_diagnostic

   A boolean flag indicating whether memory diagnostics are activated during
   simulation. Default value is ``no``.

.. inpfile:: balance_nodes

   A boolean flag indicating whether node balancing is performed during
   simulations. See also :inpfile:`balance_node_iterations` and
   :inpfile:`balance_nodes_target`.

.. inpfile:: balance_node_iterations

   The frequency at which node rebalancing is performed. Default value is ``5``.

.. inpfile:: balance_node_target

   The target balance ratio. Default value is ``1.0``.


Equation Systems
````````````````

.. inpfile:: equation_systems

   ``equation_systems`` subsection defines the physics equation sets that are
   solved for this realm and the linear solvers used to solve the different
   linear systems.

.. note::

   The variable in the :inpfile:`equation_systems` subsection are prefixed with
   ``equation_systems.name`` but only the variable name after the period should
   appear in the input file.

.. inpfile:: equation_systems.name

   A string indicating the name used in log messages etc.

.. inpfile:: equation_systems.max_iterations

   The maximum number of non-linear iterations performed during a timestep that
   couples the different equation systems.

.. inpfile:: equation_systems.solver_system_specification

   A mapping containing ``field_name: linear_solver_name`` that determines the
   linear solver used for solving the linear system. Example:

   .. code-block:: yaml

      solver_system_specification:
        pressure: solve_continuity
        enthalpy: solve_scalar
        velocity: solve_scalar

   The above example indicates that the linear systems for the enthalpy and
   momentum (velocity) equations are solved by the linear solver corresponding
   to the tag ``solve_scalar`` in the :inpfile:`linear_systems` entry, whereas
   the continuity equation system (pressure Poisson solve) should be solved
   using the linear solver definition corresponding to the tag
   ``solve_continuity``.

.. inpfile:: equation_systems.systems

   A list of equation systems to be solved within this realm. Each entry is a
   YAML mapping with the key corresponding to a pre-defined equation system name
   that contains additional parameters governing the solution of this equation
   set. The predefined equation types are

   =====================  ===========================================================
   Equation system        Description
   =====================  ===========================================================
   LowMachEOM             Low-Mach Momentum and Continuity equations
   Enthalpy               Energy equations
   ShearStressTransport   :math:`k-\omega` SST equation set
   TurbKineticEnergy      TKE equation system
   MassFraction           Mass Fraction
   MixtureFraction        Mixture Fraction
   MeshDisplacement       Arbitrary Mesh Displacement
   =====================  ===========================================================

   An example of the equation system definition for ABL precursor simulations is
   shown below:

   .. code-block:: yaml

      # Equation systems example for ABL precursor simulations
      systems:
        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1.0e-5
        - TurbKineticEnergy:
            name: myTke
            max_iterations: 1
            convergence_tolerance: 1.0e-2
        - Enthalpy:
            name: myEnth
            max_iterations: 1
            convergence_tolerance: 1.0e-2

Initial conditions
``````````````````

.. inpfile:: initial_conditions

   The ``initial_conditions`` sub-sections defines the conditions used to
   initialize the computational fields if they are not provided via the mesh
   file. Two types of field initializations are currently possible:

   - ``constant`` - Initialize the field with a constant value throughout the domain;

   - ``user_function`` - Initialize the field with a pre-defined user function.

   The snippet below shows an example of both options available to initialize
   the various computational fields used for ABL simulations. In this example,
   the pressure and turbulent kinetic energy fields are initialized using a
   constant value, whereas the velocity field is initialized by the user
   function ``boundary_layer_perturbation`` that imposes sinusoidal fluctations
   over a velocity field to trip the flow.

   .. code-block:: yaml

      initial_conditions:
        - constant: ic_1
          target_name: [fluid_part]
          value:
            pressure: 0.0
            turbulent_ke: 0.1

        - user_function: ic_2
          target_name: [fluid_part]
          user_function_name:
            velocity: boundary_layer_perturbation
          user_function_parameters:
            velocity: [1.0,0.0075398,0.0075398,50.0,8.0]

.. inpfile:: initial_conditions.constant

   This input parameter serves two purposes: 1. it indicates the *type*
   (``constant``), and 2. provides the custom *name* for this condition. In
   addition to the :inpfile:`initial_conditions.target_name` this section
   requires another entry ``value`` that contains the mapping of ``(field_name,
   value)`` as shown in the above example.

.. inpfile:: initial_conditions.user_function

   Indicates that this block of YAML input must be parsed as input for a user
   defined function.

.. inpfile:: initial_conditions.target_name

   A list of element blocks (*parts*) where this initial condition must be
   applied.

Boundary Conditions
```````````````````

.. inpfile:: boundary_conditions

   This subsection of the physics Realm contains a list of boundary conditions
   that must be used during the simulation. Each entry of this list is a YAML
   mapping entry with the key of the form ``<type>_boundary_condition`` where
   the available types are:

     - ``inflow``
     - ``open`` -- Outflow BC
     - ``wall``
     - ``symmetry``
     - ``periodic``
     - ``non_conformal`` -- e.g., BC across sliding mesh interfaces
     - ``overset`` -- overset mesh assembly description

All BC types require :inpfile:`bc.target_name` that contains a list of side sets
where the specified BC is to be applied. Additional information necessary for
certain BC types are provided by a sub-dictionary with the key
``<type>_user_data:`` that contains the parameters necessary to initialize a
specific BC type.

.. inpfile:: bc.target_name

   A list of side set part names where the given BC type must be applied. If a
   single string value is provided, it is converted to a list internally during
   input file processing phase.

Inflow Boundary Condition
+++++++++++++++++++++++++

.. code-block:: yaml

   - inflow_boundary_condition: bc_inflow
     target_name: inlet
     inflow_user_data:
       velocity: [0.0,0.0,1.0]

Open Boundary Condition
+++++++++++++++++++++++

.. code-block:: yaml

   - open_boundary_condition: bc_open
     target_name: outlet
     open_user_data:
       velocity: [0,0,0]
       pressure: 0.0

Wall Boundary Condition
+++++++++++++++++++++++

.. inpfile:: bc.wall_user_data

   This subsection contains specifications as to whether wall models are used,
   or how to treat the velocity at the wall when there is mesh motion.

The following input file snippet shows an example of using an ABL wall function at the
terrain during ABL simulations. See :ref:`theory_abl_wall_function` for more
details on the actual implementation.

.. code-block:: yaml

   # Wall boundary condition example for ABL terrain modeling
   - wall_boundary_condition: bc_terrain
     target_name: terrain
     wall_user_data:
       velocity: [0,0,0]
       use_abl_wall_function: yes
       heat_flux: 0.0
       roughness_height: 0.2
       gravity_vector_component: 3
       reference_temperature: 300.0

The entry :inpfile:`gravity_vector_component` is an integer that
specifies the component of the gravity vector, defined in
:inpfile:`solution_options.gravity`, that should be used in the
definition of the Monin-Obukhov length scale calculation.  The
entry :inpfile:`reference_temperature` is the reference temperature
used in calculation of the Monin-Obukhov length scale.

When there is mesh motion involved the wall boundary velocity takes the value of
the mesh_velocity along the part represented by :inpfile:`bc.target_name`. In
such a scenario all information under :inpfile:`bc.wall_user_data` is rendered
unused.

Example of wall boundary with a custom user function for temperature at the wall

.. code-block:: yaml

   - wall_boundary_condition: bc_6
     target_name: surface_6
     wall_user_data:
       user_function_name:
        temperature: steady_2d_thermal

Symmetry Boundary Condition
+++++++++++++++++++++++++++

Requires no additional input other than :inpfile:`bc.target_name`.

.. code-block:: yaml

   - symmetry_boundary_condition: bc_top
      target_name: top
      symmetry_user_data:


Periodic Boundary Condition
+++++++++++++++++++++++++++

Unlike the other BCs described so far, the parameter :inpfile:`bc.target_name`
behaves differently for the periodic BC. This parameter must be a list
containing exactly two entries: the boundary face pair where periodicity is
enforced. The nodes on these planes must coincide after translation in the
direction of periodicity. This BC also requires a :inpfile:`periodic_user_data`
section that specifies the search tolerance for locating node pairs.

.. inpfile:: periodic_user_data

   .. code-block:: yaml

      - periodic_boundary_condition: bc_east_west
          target_name: [east, west]
          periodic_user_data:
            search_tolerance: 0.0001

Non-Conformal Boundary
++++++++++++++++++++++

Like the periodic BC, the parameter :inpfile:`bc.target_name` must be a list
with exactly two entries that specify the boundary plane pair forming the
non-conformal boundary.

.. code-block:: yaml

   - non_conformal_boundary_condition: bc_left
     target_name: [surface_77, surface_7]
     non_conformal_user_data:
       expand_box_percentage: 10.0

Material Properties
```````````````````

.. inpfile:: material_properties

   The section provides the properties required for various physical quantities
   during the simulation. A sample section used for simulating ABL flows is shown below

   .. code-block:: yaml

      material_properties:
        target_name: [fluid_part]

        constant_specification:
         universal_gas_constant: 8314.4621
         reference_pressure: 101325.0

        reference_quantities:
          - species_name: Air
            mw: 29.0
            mass_fraction: 1.0

        specifications:
          - name: density
            type: constant
            value: 1.178037722969475
          - name: viscosity
            type: constant
            value: 1.6e-5
          - name: specific_heat
            type: constant
            value: 1000.0

.. inpfile:: material_properties.target_name

   A list of element blocks (*parts*) where the material properties are applied.
   This list should ideally include all the parts that are referenced by
   :inpfile:`initial_conditions.target_name`.

.. inpfile:: material_properties.constant_specification

   Values for several constants used during the simulation. Currently the
   following properties are defined:

   ============================ ==============================================
   Name                         Description
   ============================ ==============================================
   ``universal_gas_constant``   Ideal gas constant :math:`R`
   ``reference_temperature``    Reference temperature for simulations
   ``reference_pressure``       Reference pressure for simulations
   ============================ ==============================================

.. inpfile:: material_properties.reference_quantities

   Provides material properties for the different species involved in the
   simulation.

   ==============================  ======================================
   Name                            Description
   ==============================  ======================================
   ``species_name``                Name used to lookup properties
   ``mw``                          Molecular weight
   ``mass_fraction``               Mass fraction
   ``primary_mass_fraction``
   ``secondary_mass_fraction``
   ``stoichiometry``
   ==============================  ======================================

.. inpfile:: material_properties.specifications

   A list of material properties with the following parameters

.. inpfile:: material_properties.specifications.name

   The name used for lookup, e.g., ``density``, ``viscosity``, etc.

.. inpfile:: material_properties.specifications.type

   The type can be one of the following

   ===================== ====================================================================================
   Type                  Description
   ===================== ====================================================================================
   ``constant``          Constant value property
   ``polynomial``        Property determined by a polynomial function
   ``ideal_gas_t``       Function of :math:`T_\mathrm{ref}`, :math:`P_\mathrm{ref}`, molecular weight
   ``ideal_gas_t_p``     Function of :math:`T_\mathrm{ref}`, pressure, molecular weight
   ``ideal_gas_yk``
   ``hdf5table``         Lookup from an HDF5 table
   ``mixture_fraction``  Property determined by the mixture fraction
   ``geometric``
   ``generic``
   ===================== ====================================================================================

   **Examples**

   #. Specification for density as a function of temperature

      .. code-block:: yaml

         specifications:
            - name: density
              type: ideal_gas_t

   #. Specification of viscosity as a function of temperature

      .. code-block:: yaml

         - name: viscosity
           type: polynomial
           coefficient_declaration:
            - species_name: Air
              coefficients: [1.7894e-5, 273.11, 110.56]

      The ``species_name`` must correspond to the entry in :inpfile:`reference
      quantitites <material_properties.reference_quantities>` to lookup
      molecular weight information.

   #. Specification via ``hdf5table``

      .. code-block:: yaml

         material_properties:
           table_file_name: SLFM_CGauss_C2H4_ZMean_ZScaledVarianceMean_logChiMean.h5

           specifications:
             - name: density
               type: hdf5table
               independent_variable_set: [mixture_fraction, scalar_variance, scalar_dissipation]
               table_name_for_property: density
               table_name_for_independent_variable_set: [ZMean, ZScaledVarianceMean, ChiMean]
               aux_variables: temperature
               table_name_for_aux_variables: temperature

             - name: viscosity
               type: hdf5table
               independent_variable_set: [mixture_fraction, scalar_variance, scalar_dissipation]
               table_name_for_property: mu
               table_name_for_independent_variable_set: [ZMean, ZScaledVarianceMean, ChiMean]

   #. Specification via ``mixture_fraction``

      .. code-block:: yaml

         material_properties:
           target_name: block_1

           specifications:
             - name: density
               type: mixture_fraction
               primary_value: 0.163e-3
               secondary_value: 1.18e-3
             - name: viscosity
               type: mixture_fraction
               primary_value: 1.967e-4
               secondary_value: 1.85e-4

Mesh Motion
```````````

.. inpfile:: mesh_motion

   This subsection of the of the realm describes the rigid body motion undergone 
   by the entire mesh. The mesh motion description follows the concept of 
   reference frames in multi-body representation of physical domains, with entry 
   under :inpfile:`mesh_motion` describing a motion frame as shown below.

   Example:

   .. code-block:: yaml

      mesh_motion:
       - name: scale_background
        mesh_parts: [ Unspecified-3-HEX ]
        frame: inertial
        motion:
         - type: scaling
           factor: [1.2, 1.0, 1.2]
           origin: [5.0, 0.05, 0.0]

         - name: scale_near_body
           frame: inertial
           motion:
            - type: scaling
              factor: [1.2, 1.0, 1.2]
              origin: [0.0, 0.05, 0.0]

         - name: trans_rot_near_body
           mesh_parts: [ Unspecified-2-HEX ]
           frame: non_inertial
           reference: scale_near_body
           motion:
            - type: rotation
              omega: 12.0
              axis: [0.0, 1.0, 0.0]
              origin: [0.0, 0.05, 0.0]

            - type: translation
              start_time: 100.0
              end_time: 200.0
              velocity: [0.05, 0.0, 0.0]

.. inpfile:: mesh_motion.name

   Name of motion frame.

.. inpfile:: mesh_motion.mesh_parts

   Mesh parts associated with respective motion frame.

.. inpfile:: mesh_motion.frame

   Type of motion frame. Frames described as ``inertial`` are one time motions
   executed at the start of the simulation. Frames described as ``non_nertial``
   are executed throughout the simulation unless specified using ``start_time`` and
   ``end_time``.

.. inpfile:: mesh_motion.frame

   Name of the reference frame off of which the current motion frame is described.

.. inpfile:: mesh_motion.motion

   Type of motion the current frame undergoes. Every frame is free to undergo one
   or multiple motions simultaneously.

Output Options
``````````````

.. inpfile:: output

   Specifies the frequency of output, the output database name, etc.

   Example:

   .. code-block:: yaml

      output:
        output_data_base_name: out/ABL.neutral.e
        output_frequency: 100
        output_node_set: no
        output_variables:
         - velocity
         - pressure
         - temperature

.. inpfile:: output.output_data_base_name

   The name of the output Exodus-II database. Can specify a directory relative
   to the run directory, e.g., ``out/nalu_results.e``. The directory will be
   created automatically if one doesn't exist. Default: ``output.e``

.. inpfile:: output.output_frequency

   Nalu-Wind will write the output file every ``output_frequency`` timesteps. Note
   that currently there is no option to output results at a specified simulation
   time. Default: ``1``.

.. inpfile:: output.output_start

   Nalu-Wind will start writing output past the ``output_start`` timestep. Default: ``0``.

.. inpfile:: output.output_forced_wall_time

   Force output at a specified *wall-clock time* in seconds.

.. inpfile:: output.output_node_set

   Boolean flag indicating whether nodesets, if present, should be output to the
   output file along with element blocks.

.. inpfile:: output.compression_level

   Integer value indicating the compression level used. Default: ``0``.

.. inpfile:: output.output_variables

   A list of field names to be output to the database. The field variables can
   be node or element based quantities.


Restart Options
```````````````

.. inpfile:: restart

   This section manages the restart for this realm object.

.. inpfile:: restart.restart_data_base_name

   The filename for restart. Like :inpfile:`output`, the filename can contain a
   directory and it will be created if not already present.

.. inpfile:: restart.restart_time

   If this variable is present, it indicates that the current run will restart
   from a previous simulation. This requires that the :inpfile:`mesh` be a
   restart file with all the fields necessary for the equation sets defined in
   the :inpfile:`equation_systems.systems`. Nalu-Wind will restart from the closest
   time available in the :inpfile:`mesh` to ``restart_time``. The timesteps
   available in a restart file can be examined by looking at the ``time_whole``
   variable using the ``ncdump`` utility.

   .. note::

      The restart database used for restarting a simulation is the
      :inpfile:`mesh` parameter. The :inpfile:`restart_data_base_name
      <restart.restart_data_base_name>` parameter is used exclusively for
      outputs.

.. inpfile:: restart.restart_frequency

   The frequency at which restart files are written to the disk. Default: ``500`` timesteps.

.. inpfile:: restart.restart_start

   Nalu-Wind will write a restart file after ``restart_start`` timesteps have elapsed.

.. inpfile:: restart.restart_forced_wall_time

   Force writing of restart file after specified *wall-clock time* in seconds.

.. inpfile:: restart.restart_node_set

   A boolean flag indicating whether nodesets are output to the restart database.

.. inpfile:: restart.max_data_base_step_size

   Default: ``100,000``.

.. inpfile:: restart.compression_level

   Compression level. Default: ``0``.

Time-step Control Options
`````````````````````````

.. inpfile:: time_step_control

   This optional section specifies the adpative time stepping parameters used if
   :inpfile:`time_int.time_stepping_type` is set to ``adaptive``.

   .. code-block:: yaml

      time_step_control:
        target_courant: 2.0
        time_step_change_factor: 1.2

.. inpfile:: dtctrl.target_courant

   Maximum Courant number allowed during the simulation. Default: ``1.0``

.. inpfile:: dtctrl.time_step_change_factor

   Maximum allowable increase in ``dt`` over a given timestep.

Actuator
````````

.. inpfile:: actuator

   ``actuator`` subsection defines the inputs for actuator line simulations. A
   sample section is shown below for running actuator line simulations
   coupled to OpenFAST with two turbines.

.. code-block:: yaml

     actuator:
       type: ActLineFAST
       search_method: boost_rtree
       search_target_part: Unspecified-2-HEX

       n_turbines_glob: 2
       dry_run:  False
       debug:    False
       t_start: 0.0
       simStart: init # init/trueRestart/restartDriverInitFAST
       t_max:    5.0
       n_every_checkpoint: 100

       Turbine0:
         procNo: 0
         num_force_pts_blade: 50
         num_force_pts_tower: 20
         epsilon: [ 5.0, 5.0, 5.0 ]
         turbine_base_pos: [ 0.0, 0.0, -90.0 ]
         turbine_hub_pos: [ 0.0, 0.0, 0.0 ]
         restart_filename: ""
         FAST_input_filename: "Test01.fst"
         turb_id:  1
         turbine_name: machine_zero

       Turbine1:
         procNo: 0
         num_force_pts_blade: 50
         num_force_pts_tower: 20
         epsilon: [ 5.0, 5.0, 5.0 ]
         turbine_base_pos: [ 250.0, 0.0, -90.0 ]
         turbine_hub_pos: [ 250.0, 0.0, 0.0 ]
         restart_filename: ""
         FAST_input_filename: "Test02.fst"
         turb_id:  2
         turbine_name: machine_one


.. inpfile:: actuator.type

   Type of actuator source. Options are ``ActLineFAST`` and ``ActLinePointDrag``. Only ``ActLineFAST`` is documented here.

.. inpfile:: actuator.search_method

   String specifying the type of search method used to identify the nodes within the search radius of the actuator points. Options are ``boost_rtree`` and ``stk_kdtree``. The default is ``stk_kdtree`` when the ``search_type`` is not specified.

.. inpfile:: search_target_part

   String or an array of strings specifying the parts of the mesh to be searched to identify the nodes near the actuator points.

.. inpfile:: actuator.n_turbines_glob

   Total number of turbines in the simulation. The input file must contain a number of turbine specific sections (`Turbine0`, `Turbine1`, ..., `Turbine(n-1)`) that is consistent with `nTurbinesGlob`.

.. inpfile:: actuator.debug

   Enable debug outputs if set to true

.. inpfile:: actuator.dry_run

   The simulation will not run if dryRun is set to true. However, the simulation will read the input files, allocate turbines to processors and prepare to run the individual turbine instances. This flag is useful to test the setup of the simulation before running it.

.. inpfile:: actuator.simStart

   Flag indicating whether the simulation starts from scratch or restart. ``simStart`` takes on one of three values:

   * ``init`` - Use this option when starting a simulation from `t=0s`.
   * ``trueRestart`` - While OpenFAST allows for restart of a turbine simulation, external components like the Bladed style controller may not. Use this option when all components of the simulation are known to restart.
   * ``restartDriverInitFAST`` - When the ``restartDriverInitFAST`` option is selected, the individual turbine models start from `t=0s` and run up to the specified restart time using the inflow data stored at the actuator nodes from a hdf5 file. The C++ API stores the inflow data at the actuator nodes in a hdf5 file at every OpenFAST time step and then reads it back when using this restart option. This restart option is especially useful when the glue code is a CFD solver.

.. inpfile:: actuator.t_start

   Start time of the simulation

.. inpfile:: actuator.t_end

   End time of the simulation. ``t_end`` <= ``t_max``

.. inpfile:: actuator.t_max

   Max time of the simulation


.. note::

   ``t_max`` can only be set when OpenFAST is running from `t=0s` and ``simStart`` is ``init``. ``t_max`` can not be changed on a restart. OpenFAST will not be able to run beyond ``t_max``. Choose ``t_max`` to be large enough to accomodate any possible future extensions of runs. One can change ``t_start`` and ``t_end`` to start and stop the simulation any number of times as long as ``t_end`` <= ``t_max``.

.. inpfile:: actuator.dt_fast

   Time step for OpenFAST. All turbines should have the same time step.

.. inpfile:: actuator.n_every_checkpoint

   Restart files will be written every so many time steps

**Turbine specific input options**

.. inpfile:: actuator.turbine_base_pos

   The position of the turbine base for actuator-line simulations

.. inpfile:: actuator.num_force_pts_blade

   The number of actuator points along each blade for actuator-line simulations

.. inpfile:: actuator.num_force_pts_tower

   The number of actuator points along the tower for actuator-line simulations.

.. inpfile:: actuator.epsilon

   The spreading width :math:`\epsilon` in the Gaussian spreading function in the `[chordwise, spanwise, chord normal]` coordinate system to spread the forces from the actuator point to the nodes. Nalu-Wind currently only supports an isotropic Gaussian spreading function and uses only the value in the first component along the `chordwise` direction.

.. inpfile:: actuator.restart_filename

   The checkpoint file for this turbine when restarting a simulation

.. inpfile:: actuator.FAST_input_filename

   The FAST input file for this turbine

.. inpfile:: actuator.turb_id

   A unique turbine id for each turbine

.. include:: ./turbine_modeling.rst


Turbulence averaging
````````````````````

.. inpfile:: turbulence_averaging

   ``turbulence_averaging`` subsection defines the turbulence
   post-processing quantities and averaging procedures. A sample
   section is shown below

   .. code-block:: yaml

      turbulence_averaging:
        forced_reset: no
        time_filter_interval: 100000.0

        averaging_type: nalu_classic/moving_exponential

        specifications:

          - name: turbulence_postprocessing
	    target_name: interior
            reynolds_averaged_variables:
              - velocity

            favre_averaged_variables:
              - velocity
              - resolved_turbulent_ke

            compute_tke: yes
            compute_reynolds_stress: yes
            compute_resolved_stress: yes
            compute_temperature_resolved_flux: yes
            compute_sfs_stress: yes
            compute_temperature_sfs_flux: yes
            compute_q_criterion: yes
            compute_vorticity: yes
            compute_lambda_ci: yes

.. note::

   The variable in the :inpfile:`turbulence_averaging` subsection are
   prefixed with ``turbulence_averaging.name`` but only the variable
   name after the period should appear in the input file.

.. inpfile:: turbulence_averaging.forced_reset

   A boolean flag indicating whether the averaging of all quantities in the
   turbulence averaging section is reset. If this flag is true, the
   running average is set to zero.

.. inpfile:: turbulence_averaging.averaging_type

   This parameter sets the choice of the running average type. Possible
   values are:

   ``nalu_classic``
     "Sawtooth" average. The running average is set to zero each time the time
     filter width is reached and a new average is calculated for the next time
     interval.

   ``moving_exponential``
     "Moving window" average where the window size is set to to the time
     filter width. The contribution of any quantity before the moving window
     towards the average value reduces exponentially with every time step.

.. inpfile:: turbulence_averaging.time_filter_interval

   Number indicating the time filter size over which to calculate the
   running average. This quantity is used in different ways for each filter
   discussed above.

.. inpfile:: turbulence_averaging.specifications

   A list of turbulence postprocessing properties with the following parameters

.. inpfile:: turbulence_averaging.specifications.name

   The name used for lookup and logging.

.. inpfile:: turbulence_averaging.specifications.target_name

   A list of element blocks (parts) where the turbulence averaging is applied.

.. inpfile:: turbulence_averaging.specifications.reynolds_average_variables

   A list of field names to be averaged.

.. inpfile:: turbulence_averaging.specifications.favre_average_variables

   A list of field names to be Favre averaged.

.. inpfile:: turbulence_averaging.specifications.compute_tke

   A boolean flag indicating whether the turbulent kinetic energy is
   computed. The default value is ``no``.

.. inpfile:: turbulence_averaging.specifications.compute_reynolds_stress

   A boolean flag indicating whether the reynolds stress is
   computed. The default value is ``no``.

.. inpfile:: turbulence_averaging.specifications.compute_resolved_stress

   A boolean flag indicating whether the average resolved stress is
   computed as :math:`< \bar\rho \widetilde{u_i} \widetilde{u_j} >`.
   The default value is ``no``. When this option is turned on, the Favre
   average of the resolved velocity, :math:`< \bar{\rho} \widetilde{u_j} >`, is
   computed as well.

.. inpfile:: turbulence_averaging.specifications.compute_temperature_resolved_flux

   A boolean flag indicating whether the average resolved temperature flux is
   computed as :math:`< \bar\rho \widetilde{u_i} \widetilde{\theta} >`. The
   default value is ``no``. When this option is turned on, the Favre average
   of the resolved temperature, :math:`< \bar{\rho} \widetilde{\theta} >`, is
   computed as well.

.. inpfile:: turbulence_averaging.specifications.compute_sfs_stress

   A boolean flag indicating whether the average sub-filter scale stress is
   computed. The default value is ``no``. The sub-filter scale stress model is
   assumed to be of an eddy viscosity type and the turbulent viscosity computed
   by the turbulence model is used. The sub-filter scale kinetic energy is used
   to determine the isotropic component of the sub-filter stress. As described
   in the section :ref:`supp_eqn_set_mom_cons`, the Yoshizawa model is used to
   compute the sub-filter kinetic energy when it is not transported.

.. inpfile:: turbulence_averaging.specifications.compute_temperature_sfs_flux

   A boolean flag indicating whether the average sub-filter scale flux of
   temperature is computed. The default value is ``no``. The sub-filter scale
   stress model is assumed to be of an eddy diffusivity type and the turbulent
   diffusivity computed by the turbulence model is used along with a constant
   turbulent Prandtl number obtained from the Realm.

.. inpfile:: turbulence_averaging.specifications.compute_favre_stress

   A boolean flag indicating whether the Favre stress is computed. The
   default value is ``no``.

.. inpfile:: turbulence_averaging.specifications.compute_favre_tke

   A boolean flag indicating whether the Favre stress is computed. The
   default value is ``no``.

.. inpfile:: turbulence_averaging.specifications.compute_q_criterion

   A boolean flag indicating whether the q-criterion is computed. The
   default value is ``no``.

.. inpfile:: turbulence_averaging.specifications.compute_vorticity

   A boolean flag indicating whether the vorticity is computed. The
   default value is ``no``.

.. inpfile:: turbulence_averaging.specifications.compute_lambda_ci

   A boolean flag indicating whether the Lambda2 vorticity criterion
   is computed. The default value is ``no``.

Data probes
```````````

.. inpfile:: data_probes

   ``data_probes`` subsection defines the data probes. A sample
   section is shown below

   .. code-block:: yaml

        data_probes:

          output_frequency: 100

	  search_method: stk_octree
	  search_tolerance: 1.0e-3
	  search_expansion_factor: 2.0

	  specifications:
            - name: probe_bottomwall
	      from_target_part: bottomwall

	      line_of_site_specifications:
	        - name: probe_bottomwall
	          number_of_points: 100
		  tip_coordinates: [-6.39, 0.0, 0.0]
		  tail_coordinates: [4.0, 0.0, 0.0]

	      output_variables:
	        - field_name: tau_wall
		  field_size: 1
		- field_name: pressure

	  specifications:
            - name: probe_profile
	      from_target_part: interior

	      line_of_site_specifications:
	        - name: probe_profile
	          number_of_points: 100
		  tip_coordinates: [0, 0.0, 0.0]
		  tail_coordinates: [0.0, 0.0, 1.0]

	      output_variables:
	        - field_name: velocity
		  field_size: 3
		- field_name: reynolds_stress
		  field_size: 6

.. note::

   The variable in the :inpfile:`data_probes` subsection are prefixed
   with ``data_probes.name`` but only the variable name after the
   period should appear in the input file.

.. inpfile:: data_probes.output_frequency

   Integer specifying the frequency of output.

.. inpfile:: data_probes.search_method

   String specifying the search method for finding nodes to transfer
   field quantities to the data probe lineout.

.. inpfile:: data_probes.search_tolerance

   Number specifying the search tolerance for locating nodes.

.. inpfile:: data_probes.search_expansion_factor

   Number specifying the factor to use when expanding the node search.

.. inpfile:: data_probes.specifications

   A list of data probe properties with the following parameters

.. inpfile:: data_probes.specifications.name

   The name used for lookup and logging.

.. inpfile:: data_probes.specifications.from_target_part

   A list of element blocks (parts) where to do the data probing.

.. inpfile:: data_probes.specifications.line_of_site_specifications

   A list specifications defining the lineout

   ================= =============================================================
   Parameter         Description
   ================= =============================================================
   name              File name (without extension) for the data probe
   number_of_points  Number of points along the lineout
   tip_coordinates   List containing the coordinates for the start of the lineout
   tail_coordinates  List containing the coordinates for the end of the lineout
   ================= =============================================================


.. inpfile:: data_probes.specifications.output_variables

   A list of field names (and field size) to be probed.


Post-processing
```````````````

.. inpfile:: post_processing

   ``post_processing`` subsection defines the different
   post-processign options. A sample section is shown below

   .. code-block:: yaml

        post_processing:

	- type: surface
	  physics: surface_force_and_moment
	  output_file_name: results/wallHump.dat
	  frequency: 100
	  parameters: [0,0]
	  target_name: bottomwall

.. note::

   The variable in the :inpfile:`post_processing` subsection are prefixed with
   ``post_processing.name`` but only the variable name after the period should
   appear in the input file.

.. inpfile:: post_processing.type

   Type of post-processing. Possible values are:

   ======== ======================================
   Value    Description
   ======== ======================================
   surface  Post-processing of surface quantities
   ======== ======================================

.. inpfile:: post_processing.physics

   Physics to be post-processing. Possible values are:

   ======================================= ================================================================
   Value                                   Description
   ======================================= ================================================================
   surface_force_and_moment                Calculate surface forces and moments
   surface_force_and_moment_wall_function  Calculate surface forces and moments when using a wall function
   ======================================= ================================================================

.. inpfile:: post_processing.output_file_name

   String specifying the output file name.

.. inpfile:: post_processing.frequency

   Integer specifying the frequency of output.

.. inpfile:: post_processing.parameters

   Parameters for the physics function. For the
   ``surface_force_and_moment`` type functions, this is a
   list specifying the centroid coordinates used in the moment
   calculation.

.. inpfile:: post_processing.target_name

   A list of element blocks (parts) where to do the post-processing

.. _nalu_inp_transfers:

.. include:: ./abl_forcing.rst

Transfers
---------

.. inpfile:: transfers

   Transfers section describes the search and mapping operations to be performed
   between participating :inpfile:`realms` within a simulation.

Simulations
-----------

.. inpfile:: simulations

   This is the top-level section that orchestrates the entire execution of Nalu-Wind.
