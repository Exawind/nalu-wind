Simulations:
  - name: sim1
    time_integrator: ti_1
    optimizer: opt1

linear_solvers:

  - name: solve_scalar
    type: tpetra
    method: gmres 
    preconditioner: sgs 
    tolerance: 1e-3
    max_iterations: 75 
    kspace: 75 
    output_level: 0

realms:

  - name: realm_1
    mesh: ../../mesh/3d_split_Rot_P2_R0.g
    use_edges: no
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

      - user_function: ic_2
        target_name: [block_1, block_2]
        user_function_name:
         temperature: steady_3d_thermal

    material_properties:
      target_name: [block_1, block_2]
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

    - non_conformal_boundary_condition: bc_left
      target_name: [surface_33, surface_3]
      non_conformal_user_data:
        expand_box_percentage: 10.0 

    - non_conformal_boundary_condition: bc_right
      target_name: [surface_3, surface_33]
      non_conformal_user_data:
        expand_box_percentage: 10.0 

    solution_options:
      name: myOptions

      options:

        - element_source_terms:
            temperature: steady_3d_thermal

        - consistent_mass_matrix_png:
            temperature: no 

        - projected_nodal_gradient:
            temperature: element 

        - non_conformal:
            gauss_labatto_quadrature: no

    output:
      output_data_base_name: output/dgMMS.e
      output_frequency: 2
      output_node_set: no 
      output_variables:
       - dual_nodal_volume
       - temperature
       - dtdx
       - temperature_exact

    solution_norm:
      output_frequency: 2
      file_name: dgMMS.txt
      spacing: 12
      percision: 6
      target_name: [block_1, block_2]
      dof_user_function_pair:
       - [temperature, steady_3d_thermal]

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 10 
      time_step: 1.0
      time_stepping_type: fixed
      time_step_count: 0
      second_order_accuracy: no

      realms:
        - realm_1
