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
    muelu_xml_file_name: ../../xml/matches_ml_default.xml

transfers:

  - name: xfer_fluid_thermal
    type: geometric
    realm_pair: [realm_2, realm_1]
    mesh_part_pair: [surface_4, surface_3]
    coupling_physics: fluids_cht

  - name: xfer_thermal_fluids
    type: geometric
    realm_pair: [realm_1, realm_2]
    mesh_part_pair: [surface_3, surface_4]
    transfer_variables:
      - [temperature, temperature_bc]
      - [temperature, temperature]
     
realms:

  - name: realm_1
    mesh: ../../mesh/elbow.g
    use_edges: no
   
    boundary_conditions:

    - wall_boundary_condition: bc_exposed
      target_name: surface_1
      wall_user_data:
        heat_flux: 0.0

    - wall_boundary_condition: bc_osurface
      target_name: surface_2
      wall_user_data:
        temperature: 400

    - wall_boundary_condition: bc_isurface
      target_name: surface_3
      wall_user_data:
        reference_temperature: 300
        heat_transfer_coefficient: 0.0
        interface: yes

    - wall_boundary_condition: bc_rad
      target_name: surface_4
      wall_user_data:
        emissivity: 0.8
        irradiation: 850.0

    solution_options:
      name: myOptionsHC
      use_consolidated_face_elem_bc_algorithm: yes
      options:
        - projected_nodal_gradient:
            temperature: element
                
    initial_conditions:

      - constant: ic_1
        target_name: block_11
        value:
          temperature: 300

    material_properties:
      target_name: block_11
      specifications:
        - name: density
          type: constant
          value: 8800.0
        - name: specific_heat
          type: constant
          value: 420.0
        - name: thermal_conductivity
          type: constant
          value: 52.0

    equation_systems:
      name: theEqSys
      max_iterations: 1

      solver_system_specification:
        temperature: solve_scalar 

      systems:
        - HeatConduction:
            name: myHC
            max_iterations: 1 
            convergence_tolerance: 1.e-5

    output:
      output_data_base_name: thermal.e
      output_frequency: 5
      output_node_set: no 
      output_variables:
       - dual_nodal_volume
       - temperature
       - dtdx
       - density
       - thermal_conductivity
       - specific_heat
       - assembled_wall_area_ht
       - reference_temperature
       - heat_transfer_coefficient

  - name: realm_2
    mesh: ../../mesh/horseshoe.g
    use_edges: no

    equation_systems:
      name: theEqSys
      max_iterations: 1

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

      target_name: block_5

      constant_specification:
       universal_gas_constant: 8314.4621
       reference_pressure: 101325.0

      reference_quantities:
        - species_name: FakeAir
          mw: 28.0
          mass_fraction: 1.0

      specifications:
 
        - name: density
          type: ideal_gas_t

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
        target_name: block_5
        value:
          pressure: 0
          velocity: [0,0]  
          temperature: 300.0
  
    boundary_conditions:

    - inflow_boundary_condition: bc_inflow
      target_name: surface_1
      inflow_user_data:
        velocity: [0,0,-0.1]
        temperature: 300.0

    - open_boundary_condition: bc_side
      target_name: surface_2
      open_user_data:
        velocity: [0,0,0]
        pressure: 0.0
        temperature: 300.0

    - wall_boundary_condition: bc_nocht
      target_name: surface_3
      wall_user_data:
        velocity: [0,0,0]
        adiabatic: yes

    - wall_boundary_condition: bc_cht
      target_name: surface_4
      wall_user_data:
        velocity: [0,0,0]
        temperature: 300.0
        interface: yes 

    solution_options:
      name: myOptions
      options:
        - hybrid_factor:
            velocity: 1.0
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

    output:
      output_data_base_name: fluids.e
      output_frequency: 5
      output_node_set: no
      output_variables:
       - velocity
       - pressure
       - temperature
       - temperature_bc
       - enthalpy
       - assembled_wall_area
       - reference_temperature
       - heat_transfer_coefficient

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_time: 1.0 
      time_step: 0.025
      time_stepping_type: fixed
      time_step_count: 0
      nonlinear_iterations: 1 

      realms:
        - realm_1
        - realm_2

      transfers:
        - xfer_fluid_thermal
        - xfer_thermal_fluid
