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
    recompute_preconditioner: yes
    muelu_xml_file_name: ../../xml/matches_ml_default.xml

realms:

  - name: realm_1
    mesh: ../../mesh/100x50_P2n.g
    use_edges: no       

    equation_systems:
      name: theEqSys
      max_iterations: 4
   
      solver_system_specification:
        pressure: solve_cont
        velocity: solve_scalar
        dpdx: solve_scalar
        duidx: solve_scalar

      systems:
        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-2

    initial_conditions:

      - constant: ic_1
        target_name: block_1
        value:
          pressure: 0

      - user_function: ic_2
        target_name: block_1
        user_function_name:
         velocity: wind_energy_taylor_vortex
        user_function_parameters:
         velocity: [-2.5,0.0,0.25,15.0,10.0] 

    material_properties:
      target_name: block_1

      specifications:
        - name: density
          type: constant
          value: 1.0e-3

        - name: viscosity
          type: constant
          value: 1.0e-4

    boundary_conditions:

    - periodic_boundary_condition: bc_left_right
      target_name: [surface_1, surface_2]
      periodic_user_data:
        search_tolerance: 0.0001 

    - wall_boundary_condition: bc_top
      target_name: surface_3
      wall_user_data:
        velocity: [10.0, 0.0]

    - wall_boundary_condition: bc_bot
      target_name: surface_4
      wall_user_data:
        velocity: [10.0, 0.0]

    solution_options:
      name: myOptions
      turbulence_model: laminar
  
      options:
        - hybrid_factor:
            velocity: 0.0

        - alpha_upw:
            velocity: 1.0

        - limiter:
            pressure: no
            velocity: no

        - consistent_mass_matrix_png:
            pressure: yes
            velocity: yes

        - element_source_terms:
            momentum: [momentum_time_derivative, NSO_4TH_ALT]

    output:
      output_data_base_name: hoVortex.e
      output_frequency: 10
      output_node_set: no 
      output_variables:
       - dual_nodal_volume
       - velocity
       - pressure

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_time: 0.1
      time_step: 0.002
      time_stepping_type: fixed 
      time_step_count: 0
      second_order_accuracy: yes

      realms:
        - realm_1
