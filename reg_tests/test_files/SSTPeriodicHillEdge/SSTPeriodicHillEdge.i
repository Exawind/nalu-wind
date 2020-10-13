Simulations:
  - name: sim1
    time_integrator: ti_1
    optimizer: opt1

linear_solvers:

  - name: solve_scalar
    type: tpetra
    method: gmres
    preconditioner: riluk
    tolerance: 1e-5
    max_iterations: 200
    kspace: 200
    output_level: 0

  - name: solve_cont
    type: hypre
    method: hypre_gmres
    preconditioner: boomerAMG
    tolerance: 1e-5
    max_iterations: 50
    kspace: 75
    output_level: 0
    bamg_coarsen_type: 8
    bamg_interp_type: 6
    bamg_cycle_type: 1
    absolute_tolerance: 1.0e-8

realms:

  - name: realm_1
    mesh: ../../mesh/periodicHill.exo
    use_edges: yes
    check_for_missing_bcs: yes
    automatic_decomposition_type: rcb

    equation_systems:
      name: theEqSys
      max_iterations: 4

      solver_system_specification:
        velocity: solve_scalar
        turbulent_ke: solve_scalar
        specific_dissipation_rate: solve_scalar
        pressure: solve_cont
        ndtw: solve_cont

      systems:
        - WallDistance:
            name: myNDTW
            max_iterations: 1
            convergence_tolerance: 1.0e-8

        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1.0e-8

        - ShearStressTransport:
            name: mySST
            max_iterations: 1
            convergence_tolerance: 1.0e-8

    initial_conditions:
      - constant: ic_1
        target_name: interior-hex
        value:
          pressure: 0
          velocity: [1.0,0.0,0.0]
          turbulent_ke: 0.1
          specific_dissipation_rate: 50.0

    material_properties:
      target_name: interior-hex
      specifications:
        - name: density
          type: constant
          value: 1.0
        - name: viscosity
          type: constant
          value: 9.43396226415e-5

    boundary_conditions:

    - wall_boundary_condition: bc_top
      target_name: top
      wall_user_data:
        velocity: [0,0,0]
        turbulent_ke: 0.0
        use_wall_function: no

    - wall_boundary_condition: bc_wall
      target_name: wall
      wall_user_data:
        velocity: [0,0,0]
        turbulent_ke: 0.0
        use_wall_function: no

    - periodic_boundary_condition: bc_inlet_outlet
      target_name: [inlet, outlet]
      periodic_user_data:
        search_tolerance: 0.0001

    - periodic_boundary_condition: bc_front_back
      target_name: [front, back]
      periodic_user_data:
        search_tolerance: 0.0001

    solution_options:
      name: myOptions
      turbulence_model: sst
      projected_timescale_type: momentum_diag_inv

      fix_pressure_at_node:
       value: 0.0
       node_lookup_type: spatial_location
       location: [5.0, 1.5, 2.5]
       search_target_part: [interior-hex]
       search_method: stk_kdtree

      options:
        - hybrid_factor:
            velocity: 1.0
            turbulent_ke: 1.0
            specific_dissipation_rate: 1.0

        - alpha_upw:
            velocity: 1.0
            turbulent_ke: 1.0
            specific_dissipation_rate: 1.0

        - upw_factor:
            velocity: 1.0
            turbulent_ke: 0.0
            specific_dissipation_rate: 0.0

        - noc_correction:
            pressure: yes

        - projected_nodal_gradient:
            velocity: element
            pressure: element
            turbulent_ke: element
            specific_dissipation_rate: element
            ndtw: element

        - relaxation_factor:
            velocity: 0.7
            pressure: 0.3
            turbulent_ke: 0.7
            specific_dissipation_rate: 0.7

        - turbulence_model_constants:
            SDRWallFactor: 0.625
            periodicForcingLengthX: 4.5
            periodicForcingLengthY: 1.0
            periodicForcingLengthZ: 2.25

        - source_terms:
            momentum: body_force_box

        - source_term_parameters:
            momentum: [0.011, 0.0, 0.0]
            momentum_box: [-1.0, 1.00001, 0.0, 10.0, 4.0, 5.0]

    post_processing:

    - type: surface
      physics: surface_force_and_moment
      output_file_name: periodicHill.dat
      frequency: 1
      parameters: [0,0]
      target_name: wall

    restart:
      restart_data_base_name: restart/periodicHill.rst
      restart_frequency: 5

    output:
      output_data_base_name: results/periodicHill.e
      output_frequency: 5
      output_node_set: no
      output_variables:
       - velocity
       - density
       - pressure
       - pressure_force
       - viscous_force
       - tau_wall
       - turbulent_ke
       - specific_dissipation_rate
       - minimum_distance_to_wall
       - sst_f_one_blending
       - turbulent_viscosity

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      time_step: 4.0e-2
      termination_step_count: 5
      time_stepping_type: fixed
      time_step_count: 0
      second_order_accuracy: yes

      realms:
        - realm_1
