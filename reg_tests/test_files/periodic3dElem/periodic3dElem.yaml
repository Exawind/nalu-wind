Simulations:
  - name: sim1
    time_integrator: ti_1
    optimizer: opt1
    error_estimator: errest_1

linear_solvers:

  - name: solve_adv_diff
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
    recompute_preconditioner: no
    muelu_xml_file_name: ../../xml/milestone.xml

realms:

  - name: realm_1
    mesh: ../../mesh/periodic3d.g
    use_edges: no     
    automatic_decomposition_type: rcb

    equation_systems:
      name: theEqSys
      max_iterations: 2
   
      solver_system_specification:
        pressure: solve_cont
        velocity: solve_adv_diff
        enthalpy: solve_adv_diff

      systems:

        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-2

        - Enthalpy:
            name: myEnth
            max_iterations: 1
            convergence_tolerance: 1e-5

    initial_conditions:
      - constant: ic_1
        target_name: block_1
        value:
         velocity: [0.0,0.0,0.0]
         pressure: 0.0
         temperature: 305.0
         
    material_properties:

      target_name: block_1

      constant_specification:
       universal_gas_constant: 10000.0
       reference_pressure: 100.00

      reference_quantities:
        - species_name: N2
          mw: 30.0
          mass_fraction: 1.0

      specifications:
 
        - name: density
          type: ideal_gas_t
          
        - name: viscosity
          type: constant
          value: 1.0e-4

        - name: specific_heat
          type: constant
          value: 0.01

    boundary_conditions:

    - periodic_boundary_condition: bc_left_right
      target_name: [surface_1, surface_2]
      periodic_user_data:
        search_tolerance: 0.0001 

    - periodic_boundary_condition: bc_top_bot
      target_name: [surface_3, surface_4]
      periodic_user_data:
        search_tolerance: 0.0001 

    - periodic_boundary_condition: bc_front_back
      target_name: [surface_5, surface_6]
      periodic_user_data:
        search_tolerance: 0.0001 

    solution_options:
      name: myOptions
      turbulence_model: laminar
  
      options:

        - hybrid_factor:
            velocity: 0.0

        - limiter:
            pressure: no
            velocity: no

        - laminar_prandtl:
            enthalpy: 0.80

        - source_terms:
            momentum: buoyancy
            continuity: density_time_derivative

        - user_constants:
            gravity: [0.0,-10.0,0.0]
            reference_density: 1.0e-3

    output:
      output_data_base_name: periodic3d.e
      output_frequency: 2
      output_node_set: no 
      output_variables:
       - velocity
       - pressure
       - temperature
       - enthalpy
       - density

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 6
      time_step: 0.1
      time_stepping_type: adaptive
      time_step_count: 0
      second_order_accuracy: no

      realms:
        - realm_1
