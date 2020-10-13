Simulations:
  - name: sim1
    time_integrator: ti_1
    optimizer: opt1
    error_estimator: errest_1

linear_solvers:
  - name: solve_scalar
    type: tpetra
    method: bicgstab
    preconditioner: jacobi
    tolerance: 1e-3
    max_iterations: 200
    kspace: 200
    output_level: 0

  - name: solve_cont
    type: tpetra
    method: bicgstab
    preconditioner: muelu
    tolerance: 1e-4
    max_iterations: 200
    kspace: 200
    output_level: 0
    recompute_preconditioner: no
    muelu_xml_file_name: ../../xml/milestone.xml

realms:
  - name: realm_1
    mesh: ../../mesh/cube_64.g
    use_edges: no
    automatic_decomposition_type: rib
    matrix_free: yes
    polynomial_order: 1

    equation_systems:
      name: theEqSys
      max_iterations: 3

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
      - constant: ic_1
        target_name: block_1
        value:
          velocity: [0, 0, 0]
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
      - wall_boundary_condition: bc_1
        target_name: surface_1
        wall_user_data:
          velocity: [0, 0, 0]

      - wall_boundary_condition: bc_2
        target_name: surface_2
        wall_user_data:
          velocity: [0, 0, 0]

      - wall_boundary_condition: bc_3
        target_name: surface_3
        wall_user_data:
          velocity: [0, 0, 0]

      - wall_boundary_condition: bc_4
        target_name: surface_4
        wall_user_data:
          velocity: [0, 0, 0]

      - wall_boundary_condition: bc_5
        target_name: surface_5
        wall_user_data:
          velocity: [0, 0, 0]

      - wall_boundary_condition: bc_6
        target_name: surface_6
        wall_user_data:
          velocity: [1, 0, 0]

    solution_options:
      name: myOptions
      turbulence_model: wale
      options:
        - source_term_parameters:
            momentum: [0, 0, 1]

    output:
      output_data_base_name: cav.e
      output_frequency: 100
      output_node_set: no
      output_variables:
        - velocity
        - dpdx

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 10
      time_step: 0.015
      time_stepping_type: fixed
      time_step_count: 0
      second_order_accuracy: yes

      realms:
        - realm_1
