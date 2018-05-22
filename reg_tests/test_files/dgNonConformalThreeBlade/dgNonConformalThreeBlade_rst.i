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
    max_iterations: 50
    kspace: 50
    output_level: 0

  - name: solve_cont
    type: tpetra
    method: gmres 
    preconditioner: muelu
    tolerance: 1e-5
    max_iterations: 50
    kspace: 50
    output_level: 0
    muelu_xml_file_name: ../../xml/milestone.xml

realms:

  - name: realm_1
    mesh: dgNonConformalThreeBlade.rst
    use_edges: no       
    activate_aura: no
   
    time_step_control:
     target_courant: 2.0
     time_step_change_factor: 1.15
   
    equation_systems:
      name: theEqSys
      max_iterations: 2
   
      solver_system_specification:
        pressure: solve_cont
        velocity: solve_scalar
        mixture_fraction: solve_scalar

      systems:
        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-5

        - MixtureFraction:
            name: myZ
            max_iterations: 1
            convergence_tolerance: 1.e-2
            output_clipping_diagnostic: yes

    initial_conditions:
      - constant: ic_1
        target_name: [block_1, block_2, block_3, block_4]
        value:
          pressure: 0
          velocity: [0.5,0.0]
          mixture_fraction: 0.0

    material_properties:
      target_name: [block_1, block_2, block_3, block_4]
      specifications:
        - name: density
          type: constant
          value: 1.0

        - name: viscosity
          type: constant
          value: 1.8e-5

    boundary_conditions:

    - inflow_boundary_condition: bc_left
      target_name: surface_1
      inflow_user_data:
        velocity: [0.5,0.0,0.0]
        mixture_fraction: 0.0

    - open_boundary_condition: bc_right
      target_name: surface_2
      open_user_data:
        pressure: 0.0
        mixture_fraction: 0.0

    - symmetry_boundary_condition: bc_top
      target_name: surface_3
      symmetry_user_data:

    - symmetry_boundary_condition: bc_bot
      target_name: surface_4
      symmetry_user_data:

    - wall_boundary_condition: bc_front_blade
      target_name: surface_5
      wall_user_data:
        user_function_name:
         velocity: wind_energy
        user_function_string_parameters:
         velocity: [mmFront_ss5]
        mixture_fraction: 1.0

    - wall_boundary_condition: bc_back_lower_blade
      target_name: surface_6
      wall_user_data:
        user_function_name:
         velocity: wind_energy
        user_function_string_parameters:
         velocity: [mmBot_ss6]
        mixture_fraction: 1.0

    - wall_boundary_condition: bc_back_higher_blade
      target_name: surface_7
      wall_user_data:
        user_function_name:
         velocity: wind_energy
        user_function_string_parameters:
         velocity: [mmTop_ss7]
        mixture_fraction: 1.0

    - non_conformal_boundary_condition: bc_front_in_out
      target_name: [surface_8, surface_88]
      non_conformal_user_data:
        expand_box_percentage: 5.0 
        search_tolerance: 0.01 

    - non_conformal_boundary_condition: bc_front_out_in
      target_name: [surface_88, surface_8]
      non_conformal_user_data:
        expand_box_percentage: 5.0 
        search_tolerance: 0.01 

    - non_conformal_boundary_condition: bc_top_in_out
      target_name: [surface_9, surface_99]
      non_conformal_user_data:
        expand_box_percentage: 5.0 
        search_tolerance: 0.01 

    - non_conformal_boundary_condition: bc_top_out_in
      target_name: [surface_99, surface_9]
      non_conformal_user_data:
        expand_box_percentage: 5.0 
        search_tolerance: 0.01 

    - non_conformal_boundary_condition: bc_bot_in_out
      target_name: [surface_10, surface_1000]
      non_conformal_user_data:
        expand_box_percentage: 5.0 
        search_tolerance: 0.01 

    - non_conformal_boundary_condition: bc_bot_out_in
      target_name: [surface_1000, surface_10]
      non_conformal_user_data:
        expand_box_percentage: 5.0 
        search_tolerance: 0.01 

    solution_options:
      name: myOptions

      use_consolidated_solver_algorithm: yes
      use_consolidated_face_elem_bc_algorithm: yes

      mesh_motion:

        - name: mmBackground
          target_name: [block_1]
          omega: 0.0

        - name: mmFront_ss5
          target_name: [block_2]
          omega: 3.14
          unit_vector: [0.0,0.0,1.0]
          compute_centroid: yes

        - name: mmTop_ss7
          target_name: [block_3]
          omega: 6.28
          unit_vector: [0.0,0.0,1.0]
          compute_centroid: yes

        - name: mmBot_ss6
          target_name: [block_4]
          omega: -1.57
          unit_vector: [0.0,0.0,1.0]
          compute_centroid: yes

      options:

        - element_source_terms:
            momentum: [lumped_momentum_time_derivative, advection_diffusion, NSO_2ND_KE]
            continuity: advection
            mixture_fraction: [lumped_mixture_fraction_time_derivative, advection_diffusion, NSO_2ND_ALT]

        - non_conformal:
            gauss_labatto_quadrature: no
            upwind_advection: yes
            current_normal: yes

        - skew_symmetric_advection:
            velocity: yes
            mixture_fraction: yes

    output:
      output_data_base_name: dgNonConformalThreeBlade.e-s001
      output_frequency: 20
      output_node_set: no 
      output_forced_wall_time: 0.01
      output_variables:
       - velocity
       - pressure
       - mixture_fraction
       - mesh_displacement

    restart:
      restart_data_base_name: dgNonConformalThreeBlade.rst-s001
      restart_frequency: 25
      restart_start: 25
      restart_time: 0.0661995508335269 
      restart_forced_wall_time: 0.01

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 75 
      time_step: 0.001
      time_stepping_type: adaptive
      time_step_count: 0
      second_order_accuracy: yes

      realms:
        - realm_1
