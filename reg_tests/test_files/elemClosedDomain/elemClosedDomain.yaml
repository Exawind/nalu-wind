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
    method: cg
    preconditioner: muelu
    tolerance: 1e-5
    max_iterations: 50
    kspace: 50
    output_level: 0
    muelu_xml_file_name: ../../xml/matches_ml_default.xml
    recompute_preconditioner: no
    reuse_preconditioner: yes

realms:

  - name: realm_2
    mesh: ../../mesh/fluid_R1.g
    use_edges: no 

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

      target_name: block_2

      constant_specification:
       universal_gas_constant: 8314.4621
       reference_pressure: 101325.0

      reference_quantities:
        - species_name: FakeAir
          mw: 28.0
          mass_fraction: 1.0

      specifications:
 
        - name: density
          type: ideal_gas_t_p

        - name: viscosity
          type: polynomial
          coefficient_declaration:
           - species_name: FakeAir
             coefficients: [1.7894e-5, 273.11, 110.56]

        - name: specific_heat
          type: polynomial
          coefficient_declaration:
           - species_name: FakeAir
             low_coefficients: [3.298677000E+00, 1.408240400E-03, -3.963222000E-06, 
                                5.641515000E-09, -2.444854000E-12,-1.020899900E+03]
             high_coefficients: [3.298677000E+00, 1.408240400E-03, -3.963222000E-06, 
                                 5.641515000E-09, -2.444854000E-12,-1.020899900E+03]

    initial_conditions:
      - constant: ic_1
        target_name: block_2
        value:
          pressure: 101325.0
          velocity: [0,0]  
          temperature: 300.0
  
    boundary_conditions:

    - wall_boundary_condition: bc_bottom
      target_name: surface_1
      wall_user_data:
        velocity: [0,0]
        temperature: 1000.

    - wall_boundary_condition: bc_side
      target_name: surface_2
      wall_user_data:
        velocity: [0,0]
        adiabatic: yes

    - wall_boundary_condition: bc_top
      target_name: surface_4
      wall_user_data:
        velocity: [0,0]
        adiabatic: yes

    solution_options:
      name: myOptions

      interp_rhou_together_for_mdot: no

      options:
        - hybrid_factor:
            velocity: 1.0
            enthalpy: 1.0

        - laminar_prandtl:
            enthalpy: 1.0

        - turbulent_prandtl:
            enthalpy: 1.0

        - source_terms:
            momentum: buoyancy
            continuity: [density_time_derivative, low_speed_compressible]
            enthalpy: low_speed_compressible

        - limiter:
            pressure: no
            velocity: yes 
            enthalpy: yes

        - user_constants:
            gravity: [0.0,-9.81]
            reference_density: 0.0 

    output:
      output_data_base_name: elemClosedDomain.e 
      output_frequency: 4 
      output_node_set: no
      output_variables:
       - velocity
       - pressure
       - temperature
       - temperature_bc
       - enthalpy

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_time: 1.0
      time_step: 0.05
      time_stepping_type: fixed
      time_step_count: 0
      nonlinear_iterations: 1 

      realms:
        - realm_2
