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
    tolerance: 1e-4
    max_iterations: 100
    kspace: 100
    output_level: 0

  - name: solve_cont
    type: tpetra
    method: gmres
    preconditioner: muelu
    tolerance: 1e-4
    max_iterations: 50
    kspace: 50
    output_level: 0
    recompute_preconditioner: no
    muelu_xml_file_name: ../../xml/milestone.xml

realms:

  - name: realm_1
    mesh: ../../mesh/cube_64.g
    use_edges: yes
    automatic_decomposition_type: rib

    equation_systems:
      name: theEqSys
      max_iterations: 1

      solver_system_specification:
        pressure: solve_cont
        velocity: solve_scalar
        enthalpy: solve_scalar

      systems:
        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-12

        - Enthalpy:
            name: myEnth
            max_iterations: 1
            convergence_tolerance: 1e-12
            minimum_temperature: 50.0
            maximum_temperature: 600.0

    initial_conditions:

      - user_function: ic_1
        target_name: block_1
        user_function_name:
         velocity: BoussinesqNonIso
         temperature: BoussinesqNonIso

      - constant: ic_2
        target_name: block_1
        value:
          pressure: 0

    material_properties:
      target_name: block_1

      specifications:

        - name: density
          type: constant
          value: 1.0

        - name: viscosity
          type: constant
          value: 0.00125

        - name: specific_heat
          type: constant
          value: 0.01

    boundary_conditions:

    - inflow_boundary_condition: bc_left
      target_name: surface_1
      inflow_user_data:
        user_function_name:
         velocity: BoussinesqNonIso
         temperature: BoussinesqNonIso

    - inflow_boundary_condition: bc_right
      target_name: surface_2
      inflow_user_data:
        user_function_name:
         velocity: BoussinesqNonIso
         temperature: BoussinesqNonIso

    - inflow_boundary_condition: bc_top
      target_name: surface_3
      inflow_user_data:
        user_function_name:
         velocity: BoussinesqNonIso
         temperature: BoussinesqNonIso

    - inflow_boundary_condition: bc_bottom
      target_name: surface_4
      inflow_user_data:
        user_function_name:
         velocity: BoussinesqNonIso
         temperature: BoussinesqNonIso

    - inflow_boundary_condition: bc_front
      target_name: surface_5
      inflow_user_data:
        user_function_name:
         velocity: BoussinesqNonIso
         temperature: BoussinesqNonIso

    - inflow_boundary_condition: bc_back
      target_name: surface_6
      inflow_user_data:
        user_function_name:
         velocity: BoussinesqNonIso
         temperature: BoussinesqNonIso

    solution_options:
      name: myOptions
      turbulence_model: laminar
      options:

        - hybrid_factor:
            velocity: 0.0
            enthalpy: 0.0

        - limiter:
            pressure: no
            velocity: no
            enthalpy: no

        - laminar_prandtl:
            enthalpy: 1.0

        - source_terms:
            momentum: [buoyancy_boussinesq,BoussinesqNonIso]
            enthalpy: BoussinesqNonIso

        - user_constants:
            reference_density: 1.0
            reference_temperature: 300.0
            gravity: [0,0,-10]
            thermal_expansion_coefficient: 1.0

    solution_norm:
      output_frequency: 80
      file_name: BoussinesqNonIso_R1.dat
      spacing: 12
      percision: 6
      target_name: block_1
      dof_user_function_pair:
       - [velocity, BoussinesqNonIsoVelocity]
       - [temperature, BoussinesqNonIsoTemperature]

    output:
      output_data_base_name: BoussinesqNonIso_R1.e
      output_frequency: 20
      output_node_set: no 
      output_variables:
       - velocity
       - pressure
       - temperature
       - enthalpy
       - density
       - velocity_exact
       - temperature_exact


Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_time: 0.99999
      time_step: 0.0125
      time_stepping_type: fixed 
      time_step_count: 0
      second_order_accuracy: no

      realms:
        - realm_1
