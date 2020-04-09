Simulations:
  - name: sim1
    time_integrator: ti_1
    optimizer: opt1

linear_solvers:

 - name: solve_scalar
   type: tpetra
   method: gmres
   preconditioner: jacobi
   tolerance: 1e-7
   max_iterations: 500 
   kspace: 500
   output_level: 0

realms:

  - name: realm_1
    mesh: ../../mesh/thex8_16.g
    automatic_decomposition_type: rib
    use_edges: no
    polynomial_order: 4
    matrix_free: yes

    equation_systems:
      name: theEqSys
      max_iterations: 1

      solver_system_specification:
        temperature: solve_scalar

      systems:
        - HeatConduction:
            name: myHC
            max_iterations: 1
            convergence_tolerance: 1e-5

    initial_conditions:

      - constant: ic_1
        target_name: block_1
        value:
          temperature: 10.0

    material_properties:
      target_name: block_1
      specifications:
        - name: density
          type: constant
          value: 1000.0
        - name: thermal_conductivity
          type: constant
          value: 1.0
        - name: specific_heat
          type: constant
          value: 1.0

    boundary_conditions:

    - wall_boundary_condition: bc_1
      target_name: surface_1
      wall_user_data:
       temperature: 10

    - wall_boundary_condition: bc_2
      target_name: surface_2
      wall_user_data:
       heat_flux: 10

    solution_options:
      name: myOptions
      use_consolidated_solver_algorithm: no
      options:
        - noc_correction:
            temperature: no

    output:
      output_data_base_name: conduction_p4.e
      output_frequency: 10
      output_node_set: no 
      output_variables:
       - temperature

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 10
      time_step: 0.1
      time_stepping_type: fixed
      time_step_count: 0
      second_order_accuracy: yes

      realms:
        - realm_1
