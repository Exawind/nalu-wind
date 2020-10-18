Simulations:
  - name: sim1
    time_integrator: ti_1
    optimizer: opt1

linear_solvers:

  - name: solve_scalar
    type: tpetra
    method: gmres
    preconditioner: mt_sgs 
    tolerance: 1e-5
    max_iterations: 50
    kspace: 50
    output_level: 0

realms:

  - name: realm_1
    mesh: ../../mesh/rot_cyl_14.exo
    use_edges: no
    balance_nodes: yes
    balance_nodes_iterations: 10
    balance_nodes_target: 1.55

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
      - user_function: ic_1
        target_name: [block_1, block_2, block_3, block_4, block_5, block_6, block_7, block_8, block_9, block_10, block_11, block_12, block_13, block_14, block_15, block_16, block_17, block_18, block_19]
        user_function_name:
         temperature: steady_3d_thermal

    material_properties:
      target_name: [block_1, block_2, block_3, block_4, block_5, block_6, block_7, block_8, block_9, block_10, block_11, block_12, block_13, block_14, block_15, block_16, block_17, block_18, block_19]
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

    solution_options:
      name: myOptions

      use_consolidated_solver_algorithm: yes

      options:
 
      - projected_nodal_gradient:
          temperature: element

      - element_source_terms:
          temperature: [steady_3d_thermal, CVFEM_DIFF]

    boundary_conditions:

    - wall_boundary_condition: bc_1
      target_name: A_Inflow
      wall_user_data:
        user_function_name:
         temperature: steady_3d_thermal

    - wall_boundary_condition: bc_1
      target_name: B_Outflow
      wall_user_data:
        user_function_name:
         temperature: steady_3d_thermal

    - wall_boundary_condition: bc_3
      target_name: C_Cylinder
      wall_user_data:
        user_function_name:
         temperature: steady_3d_thermal

    - wall_boundary_condition: bc_4
      target_name: D_TopBott
      wall_user_data:
        user_function_name:
         temperature: steady_3d_thermal

    - wall_boundary_condition: bc_5
      target_name: E_Sides
      wall_user_data:
        user_function_name:
         temperature: steady_3d_thermal

    - non_conformal_boundary_condition: bc_out_in
      target_name: [G_outer, F_inner]
      non_conformal_user_data:
        expand_box_percentage: 15.0 
        search_tolerance: 0.02

    - non_conformal_boundary_condition: bc_in_out
      target_name: [F_inner, G_outer]
      non_conformal_user_data:
        expand_box_percentage: 15.0 
        search_tolerance: 0.02

    output:
      output_data_base_name: cvfemHC.e
      output_frequency: 25
      output_node_set: no 
      output_variables:
       - dual_nodal_volume
       - temperature
       - dtdx

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_time: 20.0e-3
      time_step: 2.0e-3
      time_stepping_type: fixed
      time_step_count: 0
      second_order_accuracy: yes

      realms:
        - realm_1
