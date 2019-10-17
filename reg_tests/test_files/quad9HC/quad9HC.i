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
    mesh: ../../mesh/100x50_P2n.g
    use_edges: no 

    equation_systems:
      name: theEqSys
      max_iterations: 2 
  
      solver_system_specification:
        temperature: solve_scalar
        dtdx: solve_scalar
   
      systems:
        - HeatConduction:
            name: myHC
            max_iterations: 1
            convergence_tolerance: 1e-5

    initial_conditions:

      - user_function: ic_1
        target_name: block_1
        user_function_name:
         temperature: steady_2d_thermal

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

    - periodic_boundary_condition: bc_left_right
      target_name: [surface_1, surface_2]
      periodic_user_data:
        search_tolerance: 1.e-5
        search_method: stk_kdtree

    - periodic_boundary_condition: bc_top_bottom
      target_name: [surface_3, surface_4]
      periodic_user_data:
        search_tolerance: 1.e-5
        search_method: stk_kdtree

    solution_options:
      name: myOptions

      options:

      - element_source_terms:
          temperature: steady_2d_thermal

      - projected_nodal_gradient:
          temperature: element 

      - consistent_mass_matrix_png:
          temperature: yes

    output:
      output_data_base_name: quad9HC.e
      output_frequency: 10
      output_node_set: no 
      output_variables:
       - dual_nodal_volume
       - temperature
       - dtdx

    solution_norm:
      output_frequency: 1
      file_name: quad9HC.dat
      spacing: 12
      percision: 6
      target_name: block_1
      dof_user_function_pair:
       - [temperature, steady_2d_thermal]

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 100
      time_step: 5.0 
      time_stepping_type: fixed
      time_step_count: 0
      second_order_accuracy: no

      realms:
        - realm_1
