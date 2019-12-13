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
    tolerance: 1e-14
    max_iterations: 50
    kspace: 50
    output_level: 0

  - name: solve_cont
    type: tpetra
    method: gmres
    preconditioner: muelu
    tolerance: 1e-14
    max_iterations: 100
    kspace: 100
    output_level: 0
    recompute_preconditioner: no
    muelu_xml_file_name: ../../xml/muelu_kovasznay_p7.xml

realms:

  - name: realm_1
    mesh: ../../mesh/quad4_8el.g
    use_edges: no
    polynomial_order: 5
    automatic_decomposition_type: rib     

    equation_systems:
      name: theEqSys
      max_iterations: 1

      solver_system_specification:
        pressure: solve_cont
        velocity: solve_scalar
        dpdx: solve_scalar

      systems:
        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-2

    initial_conditions:

      - user_function: icUser
        target_name: block_1
        user_function_name:
         velocity: kovasznay
         pressure: kovasznay

    material_properties:
      target_name: block_1

      specifications:

        - name: density
          type: constant
          value: 1.0

        - name: viscosity
          type: constant
          value: 0.025

    boundary_conditions:

    - inflow_boundary_condition: bc_left
      target_name: surface_1
      inflow_user_data:
        user_function_name:
          velocity: kovasznay

    - inflow_boundary_condition: bc_right
      target_name: surface_2
      inflow_user_data:
        user_function_name:
          velocity: kovasznay

    - periodic_boundary_condition: bc_top_bot
      target_name: [surface_3, surface_4]
      periodic_user_data:
        search_tolerance: 0.001

    solution_options:
      name: myOptions
      turbulence_model: laminar

      options:
        - hybrid_factor:
            velocity: 0.0

        - limiter:
            pressure: no
            velocity: no

        - element_source_terms:
            momentum: momentum_time_derivative

        - consistent_mass_matrix_png:
            pressure: yes
            velocity: no

    solution_norm:
      output_frequency: 100
      file_name: kovasznay.dat
      spacing: 12
      percision: 6
      target_name: block_1
      dof_user_function_pair:
       - [velocity, kovasznay]
       - [dpdx, kovasznay_dpdx]

    output:
      output_data_base_name: kovasznay.e
      output_frequency: 100
      output_node_set: no 
      output_variables:
       - dpdx
       - dpdx_exact
       - pressure
       - velocity
       - velocity_exact
       - dual_nodal_volume

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 100
      time_step: 0.001
      time_stepping_type: fixed
      time_step_count: 0
      second_order_accuracy: yes

      realms:
        - realm_1
