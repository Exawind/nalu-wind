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
    muelu_xml_file_name: ../../xml/milestone.xml

transfers:

# Fluid to ....

  - name: xfer_fluid_io
    type: geometric
    realm_pair: [fluidRealm, ioRealm]
    mesh_part_pair: [Front, block_101]
    objective: input_output
    transfer_variables:
      - [velocity, velocity_bc]
      - [velocity, cont_velocity_bc]
      - [temperature, temperature_bc]

realms:

  - name: fluidRealm
    mesh: ../../mesh/abl_1km_cube_toy.g
    use_edges: yes
    automatic_decomposition_type: rcb

    equation_systems:
      name: theEqSys
      max_iterations: 4

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

      target_name: Unspecified-2-HEX

      constant_specification:
       universal_gas_constant: 8314.4621
       reference_pressure: 101325.0

      reference_quantities:
        - species_name: Air
          mw: 29.0
          mass_fraction: 1.0

      specifications:
 
        - name: density
          type: ideal_gas_t

        - name: viscosity
          type: polynomial
          coefficient_declaration:
           - species_name: Air
             coefficients: [1.7894e-5, 273.11, 110.56]

        - name: specific_heat
          type: polynomial
          coefficient_declaration:
           - species_name: Air
             low_coefficients: [3.298677000E+00, 1.408240400E-03, -3.963222000E-06, 
                                5.641515000E-09, -2.444854000E-12,-1.020899900E+03]
             high_coefficients: [3.298677000E+00, 1.408240400E-03, -3.963222000E-06, 
                                 5.641515000E-09, -2.444854000E-12,-1.020899900E+03]

    initial_conditions:
      - constant: ic_1
        target_name: Unspecified-2-HEX
        value:
          pressure: 0
          temperature: 300.0
          velocity: [10.0, 0.0, 0.0]

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
        velocity: [10.0,0,0]
        pressure: 0.0
        temperature: 300.0

    - wall_boundary_condition: bc_lower
      target_name: Ground
      wall_user_data:
        velocity: [0,0,0]
        use_abl_wall_function: yes
        heat_flux: 301.5
        reference_temperature: 300.0
        roughness_height: 0.1
        gravity_vector_component: 3

    solution_options:
      name: myOptions
      turbulence_model: wale
      interp_rhou_together_for_mdot: yes

      options:

        - laminar_prandtl:
            enthalpy: 0.7

        - turbulent_prandtl:
            enthalpy: 1.0

        - source_terms:
            momentum: buoyancy
            continuity: density_time_derivative

        - user_constants:
            gravity: [0.0,0.0,-9.81]
            reference_density: 1.2

        - hybrid_factor:
            velocity: 0.0
            enthalpy: 1.0

        - limiter:
            pressure: no
            velocity: no
            enthalpy: yes 

        - peclet_function_form:
            velocity: tanh
            enthalpy: tanh

        - peclet_function_tanh_transition:
            velocity: 5000.0
            enthalpy: 2.01

        - peclet_function_tanh_width:
            velocity: 200.0
            enthalpy: 4.02

        - source_terms:
            momentum: body_force

        - source_term_parameters:
            momentum: [0.000135, 0.0, 0.0]

    data_probes:

      output_frequency: 1
      search_method: stk_kdtree
      search_tolerance: 1.0e-3
      search_expansion_factor: 2.0

      lidar_specifications:
        from_target_part: [Unspecified-2-HEX]
        inner_prism_initial_theta: 90
        inner_prism_rotation_rate: 3.5
        inner_prism_azimuth: 15.2
        outer_prism_initial_theta: 90
        outer_prism_rotation_rate: 6.5
        outer_prism_azimuth: 15.2
        scan_time: 2 #seconds
        number_of_samples: 984
        points_along_line: 100
        center: [500,500,100]
        beam_length: 1.0
        axis: [1,0,0]
        ground_direction: [0,0,1]


    output:
      output_data_base_name: abl_1km_cube.e
      output_frequency: 5
      output_node_set: no
      output_variables:
       - velocity
       - pressure
       - enthalpy
       - temperature
       - specific_heat
       - viscosity

  - name: ioRealm
    mesh: ../../mesh/abl_io.g
    type: input_output
    automatic_decomposition_type: rcb

    field_registration:
      specifications:

        - field_name: velocity_bc
          target_name: block_101
          field_size: 3
          field_type: node_rank

        - field_name: cont_velocity_bc
          target_name: block_101
          field_size: 3
          field_type: node_rank

        - field_name: temperature_bc
          target_name: block_101
          field_size: 1
          field_type: node_rank

    output:
      output_data_base_name: abl_io_results.e
      output_frequency: 1
      output_node_set: no
      output_variables:
       - velocity_bc
       - cont_velocity_bc
       - temperature_bc

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 10
      time_step: 0.5
      time_stepping_type: fixed
      time_step_count: 0
      second_order_accuracy: yes

      realms:
        - fluidRealm
        - ioRealm
