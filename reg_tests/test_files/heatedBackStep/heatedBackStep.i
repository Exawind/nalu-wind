Simulations:
  - name: sim1
    time_integrator: ti_1
    optimizer: opt1

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
    recompute_preconditioner: no
    muelu_xml_file_name: ../../xml/milestone.xml

realms:

  - name: realm_1
    mesh: ../../mesh/heatedBackStep.g
    use_edges: no 

    time_step_control:
     target_courant: 2.0
     time_step_change_factor: 1.25

    equation_systems:
      name: theEqSys
      max_iterations: 2

      solver_system_specification:
        velocity: solve_scalar
        pressure: solve_cont
        enthalpy: solve_scalar

      systems:

        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-5

        - Enthalpy:
            name: myEnth
            max_iterations: 1
            convergence_tolerance: 1e-5

    material_properties:

      target_name: block_1

      constant_specification:
       universal_gas_constant: 8314.4621
       reference_pressure: 101325.0

      reference_quantities:
        - species_name: N2
          mw: 14.0
          mass_fraction: 0.767

        - species_name: O2
          mw: 32.0
          mass_fraction: 0.233

      specifications:
 
        - name: density
          type: ideal_gas_t

        - name: viscosity
          type: polynomial
 
          coefficient_declaration:
          - species_name: O2 
            coefficients: [1.7894e-5, 273.11, 110.56]

          - species_name: N2 
            coefficients: [1.7894e-5, 273.11, 110.56]

        - name: specific_heat
          type: polynomial

          coefficient_declaration:
          - species_name: O2 
            low_coefficients: [3.212936000E+00,  1.127486400E-03, -5.756150000E-07,   
                               1.313877300E-09, -8.768554000E-13, -1.005249000E+03]
            high_coefficients: [3.212936000E+00,  1.127486400E-03, -5.756150000E-07,   
                                1.313877300E-09, -8.768554000E-13, -1.005249000E+03]

          - species_name: N2 
            low_coefficients: [3.298677000E+00,  1.408240400E-03, -3.963222000E-06, 
                               5.641515000E-09, -2.444854000E-12, -1.020899900E+03]
            high_coefficients: [3.298677000E+00,  1.408240400E-03, -3.963222000E-06, 
                                5.641515000E-09, -2.444854000E-12, -1.020899900E+03]

    initial_conditions:
      - constant: ic_1
        target_name: block_1
        value:
          pressure: 0
          velocity: [0,0]  
          temperature: 298.0
  
    boundary_conditions:

    - inflow_boundary_condition: bc_inflow
      target_name: surface_1
      inflow_user_data:
        velocity: [2.0,0,0.0]
        temperature: 298.0

    - open_boundary_condition: bc_open
      target_name: surface_2
      open_user_data:
        velocity: [0,0,0]
        pressure: 0.0
        temperature: 298.0

    - wall_boundary_condition: bc_first_bottom
      target_name: surface_3
      wall_user_data:
        velocity: [0,0,0]
        heat_flux: 0.0

    - wall_boundary_condition: bc_vertical
      target_name: surface_4
      wall_user_data:
        velocity: [0,0,0]
        heat_flux: 0.0

    - wall_boundary_condition: bc_formerFuel
      target_name: surface_9
      wall_user_data:
        velocity: [0,0,0]
        heat_flux: 1000.0

    - wall_boundary_condition: bc_bottomHeated
      target_name: surface_5
      wall_user_data:
        velocity: [0,0,0]
        heat_flux: 0.0

    - periodic_boundary_condition: bc_left_right
      target_name: [surface_6, surface_7]
      periodic_user_data:
        search_tolerance: 1.e-5
        search_method: stk_kdtree

    - wall_boundary_condition: bc_top
      target_name: surface_8
      wall_user_data:
        velocity: [0, 0, 0]
        heat_flux: 0.0

    solution_options:
      name: myOptions
      options:
        - hybrid_factor:
            velocity: 0.0
            enthalpy: 1.0

        - laminar_prandtl:
            enthalpy: 1.0

        - turbulent_prandtl:
            enthalpy: 1.0

        - source_terms:
            continuity: density_time_derivative

        - limiter:
            pressure: no
            velocity: no
            enthalpy: yes 

        - projected_nodal_gradient:
            pressure: element 
            velocity: element 
            enthalpy: element 

    output:
      output_data_base_name: heatedBackStep.e
      output_frequency: 10
      output_node_set: no
      output_variables:
       - velocity
       - pressure
       - temperature
       - enthalpy

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 31 
      time_step: 1.0e-3
      time_stepping_type: adaptive
      time_step_count: 0
      second_order_accuracy: yes

      realms:
        - realm_1
