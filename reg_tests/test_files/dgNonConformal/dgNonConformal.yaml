Simulations:
  - name: sim1
    time_integrator: ti_1
    optimizer: opt1

linear_solvers:

  - name: solve_scalar
    type: tpetra
    method: biCgStab 
    preconditioner: sgs 
    tolerance: 1e-3
    max_iterations: 75 
    kspace: 75 
    output_level: 0

realms:

  - name: realm_1
    mesh: ../../mesh/2DTwoBlock_R0_rev.g
    use_edges: no

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
         temperature: steady_2d_thermal

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
         temperature: steady_2d_thermal

    - wall_boundary_condition: bc_2
      target_name: surface_2
      wall_user_data:
        user_function_name:
         temperature: steady_2d_thermal

    - wall_boundary_condition: bc_3
      target_name: surface_3
      wall_user_data:
        user_function_name:
         temperature: steady_2d_thermal

    - wall_boundary_condition: bc_4
      target_name: surface_4
      wall_user_data:
        user_function_name:
         temperature: steady_2d_thermal

    - wall_boundary_condition: bc_5
      target_name: surface_5
      wall_user_data:
        user_function_name:
         temperature: steady_2d_thermal

    - wall_boundary_condition: bc_6
      target_name: surface_6
      wall_user_data:
        user_function_name:
         temperature: steady_2d_thermal

    - non_conformal_boundary_condition: bc_left
      target_name: [surface_77, surface_7]
      non_conformal_user_data:
        expand_box_percentage: 10.0 

    - non_conformal_boundary_condition: bc_right
      target_name: [surface_7, surface_77]
      non_conformal_user_data:
        expand_box_percentage: 10.0 

    solution_options:
      name: myOptions

      options:

        - source_terms:
            temperature: steady_2d_thermal

        - projected_nodal_gradient:
            temperature: element 

        - non_conformal:
            gauss_labatto_quadrature: yes
            activate_coincident_node_error_check: yes

    output:
      output_data_base_name: dgNonConformal.e
      output_frequency: 100 
      output_node_set: no 
      output_variables:
       - dual_nodal_volume
       - temperature
       - temperature_exact
       - dtdx

    solution_norm:
      output_frequency: 10
      file_name: theNorm.dat
      spacing: 12
      percision: 6
      target_name: [block_1, block_2]
      dof_user_function_pair:
       - [temperature, steady_2d_thermal]

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 200 
      time_step: 10.0
      time_stepping_type: fixed
      time_step_count: 0
      second_order_accuracy: no

      realms:
        - realm_1
