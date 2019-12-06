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

realms:

  - name: realm_1
    mesh: tamsChannelEdge.rst
    use_edges: yes 
    check_for_missing_bcs: yes
    support_inconsistent_multi_state_restart: yes

    time_step_control:
     target_courant: 1.0
     time_step_change_factor: 1.2

    equation_systems:
      name: theEqSys
      max_iterations: 4

      solver_system_specification:
        velocity: solve_scalar
        turbulent_ke: solve_scalar
        specific_dissipation_rate: solve_scalar
        pressure: solve_cont
        ndtw: solve_cont
        time_averaged_model_split: solve_cont

      systems:
        - WallDistance:
            name: myNDTW
            max_iterations: 1
            convergence_tolerance: 1.0e-8

        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-8

        - ShearStressTransport:
            name: mySST
            max_iterations: 1
            convergence_tolerance: 1e-8

    initial_conditions:
      - constant: ic_1
        target_name: Unspecified-2-HEX
        value:
          pressure: 0
          velocity: [22.0,0.0,0.0]
          turbulent_ke: 0.05
          specific_dissipation_rate: 3528.0
          average_velocity: [22.78,0.0,0.0]
          average_tke_resolved: 0.0
          average_dudx: 0.0
          k_ratio: 1.0
          avg_res_adequacy_parameter: 1.0 

    material_properties:
      target_name: Unspecified-2-HEX
      specifications:
        - name: density
          type: constant
          value: 1.0
        - name: viscosity
          type: constant
          value: 9.99488e-4 

    boundary_conditions:

    - wall_boundary_condition: bc_bot
      target_name: bottom
      wall_user_data:
        velocity: [0,0,0]
        turbulent_ke: 0.0
        use_wall_function: no

    - wall_boundary_condition: bc_top
      target_name: top
      wall_user_data:
        velocity: [0,0,0]
        turbulent_ke: 0.0
        use_wall_function: no

    - periodic_boundary_condition: bc_inlet_outlet
      target_name: [inlet, outlet]
      periodic_user_data:
        search_tolerance: 0.001

    - periodic_boundary_condition: bc_front_back
      target_name: [front, back]
      periodic_user_data:
        search_tolerance: 0.001

    solution_options:
      name: myOptions
      turbulence_model: sst_tams
      reset_TAMS_averages_on_init: false 
      projected_timescale_type: momentum_diag_inv

      fix_pressure_at_node:
       value: 0.0
       node_lookup_type: spatial_location
       location: [ 1.0, 1.0, 1.0 ]
       search_target_part: [Unspecified-2-HEX]
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

        - limiter:
            pressure: no
            velocity: yes
            turbulent_ke: yes
            specific_dissipation_rate: yes

        - projected_nodal_gradient:
            velocity: element
            pressure: element
            turbulent_ke: element
            specific_dissipation_rate: element
            ndtw: element

        - relaxation_factor:
            velocity: 1.0 
            pressure: 1.0
            turbulent_ke: 1.0
            specific_dissipation_rate: 1.0

        - turbulence_model_constants:
            SDRWallFactor: 0.625
            forcingFactor: 32.0

        - source_terms:
            momentum: body_force

        - source_term_parameters:
            momentum: [1.00, 0.0, 0.0]

    restart:
      restart_data_base_name: tamsChannelEdge.rst-s001
      restart_frequency: 10
      restart_start: 5
      restart_time: 100

    output:
      output_data_base_name: tamsChannelEdge.e-s001
      output_frequency: 10
      output_node_set: no
      output_variables:
       - velocity
       - average_velocity
       - pressure
       - pressure_force
       - tau_wall
       - turbulent_ke
       - specific_dissipation_rate
       - minimum_distance_to_wall
       - turbulent_viscosity
       - k_ratio
       - rans_time_scale
       - average_mass_flow_rate
       - average_tke_resolved
       - avg_res_adequacy_parameter
       - resolution_adequacy_parameter
       - metric_tensor
       - element_courant
       - average_production
       - average_dudx

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      time_step: 2.0e-3
      termination_step_count: 20
      time_stepping_type: fixed
      time_step_count: 0
      second_order_accuracy: yes

      realms:
        - realm_1
