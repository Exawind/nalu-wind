Simulations:
  - name: sim1
    time_integrator: ti_1
    optimizer: opt1
    error_estimator: errest_1

linear_solvers:

  - name: solve_scalar
    type: tpetra
    method: gmres
    preconditioner: sgs 
    tolerance: 1e-6
    max_iterations: 1000
    kspace: 1000
    output_level: 0

  - name: solve_cont
    type: tpetra
    method: gmres
    preconditioner: muelu
    tolerance: 1e-6
    max_iterations: 1000
    kspace: 1000
    output_level: 0
    recompute_preconditioner: yes
    muelu_xml_file_name: ../../xml/milestone.xml

realms:

  - name: realm_1
    mesh: ../../mesh/hex8_32.g
    use_edges: no
    automatic_decomposition_type: rib

    equation_systems:
      name: theEqSys
      max_iterations: 4

      solver_system_specification:
        pressure: solve_cont
        velocity: solve_scalar
        dpdx: solve_scalar

      systems:
        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-4

    initial_conditions:

      - user_function: ic_1
        target_name: block_1
        user_function_name:
         velocity: BoussinesqNonIso

      - constant: ic_2
        target_name: block_1
        value:
          pressure: 0

    material_properties:
      target_name: block_1

      specifications:

        - name: density
          type: constant
          value: 1.0

        - name: viscosity
          type: constant
          value: 1.0e-3

    boundary_conditions:

    - inflow_boundary_condition: bc_left
      target_name: surface_1
      inflow_user_data:
        user_function_name:
          velocity: BoussinesqNonIso

    - inflow_boundary_condition: bc_right
      target_name: surface_2
      inflow_user_data:
        user_function_name:
          velocity: BoussinesqNonIso

    - inflow_boundary_condition: bc_top
      target_name: surface_3
      inflow_user_data:
        user_function_name:
          velocity: BoussinesqNonIso

    - inflow_boundary_condition: bc_bottom
      target_name: surface_4
      inflow_user_data:
        user_function_name:
          velocity: BoussinesqNonIso

    - inflow_boundary_condition: bc_front
      target_name: surface_5
      inflow_user_data:
        user_function_name:
          velocity: BoussinesqNonIso

    - inflow_boundary_condition: bc_back
      target_name: surface_6
      inflow_user_data:
        user_function_name:
         velocity: BoussinesqNonIso

    solution_options:
      name: myOptions
      turbulence_model: laminar

      use_consolidated_solver_algorithm: yes
      use_consolidated_face_elem_bc_algorithm: yes
      projected_timescale_type: default

      options:
        - element_source_terms:
            momentum: [lumped_momentum_time_derivative, advection_diffusion, tgmms]
            continuity: [advection]

        - consistent_mass_matrix_png:
            pressure: no

    solution_norm:
      output_frequency: 25
      file_name: tg_mms_weak_R0.dat
      spacing: 12
      percision: 6
      target_name: block_1
      dof_user_function_pair:
       - [velocity, BoussinesqNonIsoVelocity]

    output:
      output_data_base_name: solution/tg_mms_weak_R0.e
      output_frequency: 25
      output_node_set: no 
      output_variables:
       - velocity
       - dpdx
       - velocity_exact

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_time: 0.1
      time_step: 0.02
      time_stepping_type: fixed 
      time_step_count: 0
      second_order_accuracy: yes

      realms:
        - realm_1
