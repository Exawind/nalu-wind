Actuator Turbine Model
``````````````````````

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
         nacelle_cd: 1.0
         nacelle_area: 1.0
         air_density: 1.225
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
         nacelle_cd: 1.0
         nacelle_area: 1.0
         air_density: 1.225
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

.. inpfile:: actuator.nacelle_cd

   The drag coefficient for the nacelle. If this is set to zero, or not
   defined, the code will not implement the nacelle model.

.. inpfile:: actuator.nacelle_area

   The reference area for the nacelle. This is only used if the nacelle
   model is used.

.. inpfile:: actuator.air_density

   The air density. This is only used to compute the nacelle force. It should
   match the density being used in both Nalu and OpenFAST.

.. inpfile:: actuator.epsilon

   The spreading width :math:`\epsilon` in the Gaussian spreading function in the `[chordwise, spanwise, chord normal]` coordinate system to spread the forces from the actuator point to the nodes. Nalu currently only supports an isotropic Gaussian spreading function and uses only the value in the first component along the `chordwise` direction.

.. inpfile:: actuator.restart_filename

   The checkpoint file for this turbine when restarting a simulation

.. inpfile:: actuator.FAST_input_filename

   The FAST input file for this turbine

.. inpfile:: actuator.turb_id

   A unique turbine id for each turbine
