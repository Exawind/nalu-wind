Simulations:
  - name: sim1
    time_integrator: ti_1
    optimizer: opt1

linear_solvers:

  - name: solve_scalar
    type: tpetra
    method: gmres 
    preconditioner: muelu
    tolerance: 1e-3
    max_iterations: 75 
    kspace: 75 
    output_level: 0
    muelu_xml_file_name: ../../xml/milestone.xml

realms:

  - name: realm_1
    mesh: ../../mesh/thex8_4.g
    use_edges: no
    polynomial_order: 3
    automatic_decomposition_type: rcb

    equation_systems:
      name: theEqSys
      max_iterations: 2 

      solver_system_specification:
        temperature: solve_scalar

      systems:
        - HeatConduction:
            name: myHC
            max_iterations: 1
            convergence_tolerance: 1e-5

    initial_conditions:

      - user_function: ic_1
        target_name: block_1
        user_function_name:
         temperature: steady_3d_thermal

    material_properties:
      target_name: block_1
      specifications:
        - name: density
          type: constant
          value: 1.0
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
        user_function_name:
         temperature: steady_3d_thermal

    - wall_boundary_condition: bc_2
      target_name: surface_2
      wall_user_data:
        user_function_name:
         temperature: steady_3d_thermal

    - wall_boundary_condition: bc_3
      target_name: surface_3
      wall_user_data:
        user_function_name:
         temperature: steady_3d_thermal

    - wall_boundary_condition: bc_4
      target_name: surface_4
      wall_user_data:
        user_function_name:
         temperature: steady_3d_thermal

    - wall_boundary_condition: bc_5
      target_name: surface_5
      wall_user_data:
        user_function_name:
         temperature: steady_3d_thermal

    - wall_boundary_condition: bc_6
      target_name: surface_6
      wall_user_data:
        user_function_name:
         temperature: steady_3d_thermal

    solution_options:
      name: myOptions

      use_consolidated_solver_algorithm: no

      options:
        - element_source_terms:
            temperature: [steady_3d_thermal]

    output:
      output_data_base_name: cvfemHexHC_P3_R0.e
      output_frequency: 10
      output_node_set: no 
      output_variables:
       - dual_nodal_volume
       - temperature

    solution_norm:
      output_frequency: 10
      file_name: cvfemHexHC_P3_R0.dat
      spacing: 12
      percision: 6
      target_name: block_1
      dof_user_function_pair:
        - [temperature, steady_3d_thermal]

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 10
      time_step: 5.0 
      time_stepping_type: fixed
      time_step_count: 0
      second_order_accuracy: yes

      realms:
        - realm_1
