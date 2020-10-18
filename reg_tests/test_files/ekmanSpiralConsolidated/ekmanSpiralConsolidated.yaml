Simulations:
  - name: sim1
    time_integrator: ti_1
    optimizer: opt1

linear_solvers:

  - name: solve_scalar
    type: tpetra
    method: gmres
    preconditioner: ilut
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
    mesh: ../../mesh/ekmanSpiral.g
    use_edges: no
    automatic_decomposition_type: rcb

    time_step_control:
      target_courant: 10.0
      time_step_change_factor: 1.2

    equation_systems:
      name: theEqSys
      max_iterations: 2

      solver_system_specification:
        velocity: solve_scalar
        pressure: solve_cont

      systems:

        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-5

    material_properties:

      target_name: Unspecified-2-HEX

      specifications:
 
        - name: density
          type: constant
          value: 1.0

        - name: viscosity
          type: constant
          value: 5.0

    initial_conditions:
      - constant: ic_1
        target_name: Unspecified-2-HEX
        value:
          pressure: 0
          velocity: [15.0, 0.0, 0.0]

    boundary_conditions:

    - periodic_boundary_condition: bc_left_right
      target_name: [Front, Back]
      periodic_user_data:
        search_tolerance: 0.0001

    - periodic_boundary_condition: bc_front_back
      target_name: [Left, Right]
      periodic_user_data:
        search_tolerance: 0.0001 

    - open_boundary_condition: bc_open
      target_name: Top
      open_user_data:
        velocity: [15.0,0,0]
        pressure: 0.0

    - wall_boundary_condition: bc_lower
      target_name: Ground
      wall_user_data:
        velocity: [0,0,0]

    solution_options:
      name: myOptions
      turbulence_model: laminar
      use_consolidated_solver_algorithm: yes

      options:

        - element_source_terms:
            momentum: [momentum_time_derivative, advection_diffusion, EarthCoriolis]
            continuity: [advection]

        - user_constants:
            east_vector: [1.0,0.0,0.0]
            north_vector: [0.0,1.0,0.0]
            latitude: 30.0
            earth_angular_velocity: 7.2921159e-5

        - hybrid_factor:
            velocity: 1.0

        - limiter:
            pressure: no
            velocity: no

        - source_terms:
            momentum: body_force

        - source_term_parameters:
            momentum: [0.0, 0.0010935, 0.0]

    output:
      output_data_base_name: ekmanSpiralConsolidated.e
      output_frequency: 2 
      output_node_set: no 
      output_variables:
       - velocity
       - pressure

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 20
      time_stepping_type: adaptive
      time_step_count: 0
      second_order_accuracy: no

      realms:
        - realm_1
