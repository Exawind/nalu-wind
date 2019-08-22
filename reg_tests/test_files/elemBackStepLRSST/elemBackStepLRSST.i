Simulations:
  - name: sim1
    time_integrator: ti_1
    optimizer: opt1

linear_solvers:

  - name: solve_scalar
    type: tpetra
    method: gmres
    preconditioner: sgs
    tolerance: 1e-3
    max_iterations: 50
    kspace: 50
    output_level: 0

  - name: solve_cont
    type: tpetra
    method: gmres
    preconditioner: muelu
    tolerance: 1e-3
    max_iterations: 50
    kspace: 50
    output_level: 0
    muelu_xml_file_name: ../../xml/matches_ml_default.xml

transfers:

# Fluid to ....

  - name: xfer_fluid_io_volume
    type: geometric
    realm_pair: [fluidRealm, ioRealm]
    mesh_part_pair: [block_1, block_1]
    objective: input_output
    search_tolerance: 5.0e-4
    transfer_variables:
      - [minimum_distance_to_wall, minimum_distance_to_wall]
      - [dudx, dudx]

  - name: xfer_fluid_io_surface
    type: geometric
    realm_pair: [fluidRealm, ioRealm]
    from_target_name: [surface_4, surface_5, surface_6]
    to_target_name: [surface_4, surface_5, surface_6]
    objective: input_output
    search_expansion_factor: 1.45
    transfer_variables:
      - [yplus, yplus]

realms:

  - name: fluidRealm
    mesh: ../../mesh/backstep_miny.g
    use_edges: no

    time_step_control:
     target_courant: 25.0
     time_step_change_factor: 1.2

    equation_systems:
      name: theEqSys
      max_iterations: 1 

      solver_system_specification:
        velocity: solve_scalar
        turbulent_ke: solve_scalar
        specific_dissipation_rate: solve_scalar
        pressure: solve_cont

      systems:

        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-5

        - ShearStressTransport:
            name: mySST 
            max_iterations: 1
            convergence_tolerance: 1e-5

    initial_conditions:
      - constant: ic_1
        target_name: block_1
        value:
          pressure: 0
          velocity: [796.6,0.0,0.0]
          turbulent_ke: 9518.6
          specific_dissipation_rate: 1084.0

    material_properties:
      target_name: block_1
      specifications:
        - name: density
          type: constant
          value: 1.138e-3
        - name: viscosity
          type: constant
          value: 1.813e-4

    boundary_conditions:

    - inflow_boundary_condition: bc_inflow
      target_name: surface_1
      inflow_user_data:
        velocity: [796.6,0.0, 0.0]
        turbulent_ke: 9518.6
        specific_dissipation_rate: 1084.0

    - open_boundary_condition: bc_open
      target_name: surface_2
      open_user_data:
        velocity: [0,0]
        pressure: 0.0
        turbulent_ke: 1.0e-12
        specific_dissipation_rate: 1.0e-6

    - symmetry_boundary_condition: bc_symBottom
      target_name: surface_3
      symmetry_user_data:

    - wall_boundary_condition: bc_wall
      target_name: surface_4
      wall_user_data:
        velocity: [0,0,0]
        turbulent_ke: 0.0
        use_wall_function: no

    - wall_boundary_condition: bc_top
      target_name: surface_5
      wall_user_data:
        velocity: [0,0,0]
        turbulent_ke: 0.0
        use_wall_function: no

    - wall_boundary_condition: bc_step
      target_name: surface_6
      wall_user_data:
        velocity: [0,0,0]
        turbulent_ke: 0.0
        use_wall_function: no

    - symmetry_boundary_condition: bc_sym
      target_name: surface_7
      symmetry_user_data:

    - symmetry_boundary_condition: bc_sym
      target_name: surface_8
      symmetry_user_data:

    solution_options:
      name: myOptions
      turbulence_model: sst_des

      options:
        - hybrid_factor:
            velocity: 1.0 
            turbulent_ke: 1.0
            specific_dissipation_rate: 1.0

        - alpha_upw:
            velocity: 1.0 

        - limiter:
            pressure: no
            velocity: yes 
            turbulent_ke: yes
            specific_dissipation_rate: yes

        - projected_nodal_gradient:
            velocity: element 
            pressure: element
            turbulent_ke: element 
            specific_dissipation_rate: element 

        - input_variables_from_file:
            minimum_distance_to_wall: ndtw

    output:
      output_data_base_name: elemBackStepLRSST.e
      output_frequency: 5
      output_node_set: no 
      output_variables:
       - velocity
       - pressure
       - pressure_force
       - tau_wall
       - turbulent_ke
       - specific_dissipation_rate
       - minimum_distance_to_wall
       - sst_f_one_blending
       - sst_max_length_scale
       - turbulent_viscosity

    post_processing:

    - type: surface
      physics: surface_force_and_moment
      output_file_name: post_process_one.dat
      frequency: 1
      parameters: [0,0,0]
      target_name: [surface_4, surface_5, surface_6]

  - name: ioRealm
    mesh: ../../mesh/backstep_miny.g
    type: input_output

    field_registration:
      specifications:
        - field_name: dudx
          target_name: block_1
          field_size: 9
          field_type: node_rank

        - field_name: minimum_distance_to_wall
          target_name: block_1
          field_size: 1
          field_type: node_rank

        - field_name: yplus
          target_name: [surface_4, surface_5, surface_6]
          field_size: 1
          field_type: node_rank

    output:
      output_data_base_name: IO_subset.e
      output_frequency: 2
      output_node_set: no
      output_variables:
       - dudx
       - yplus
       - minimum_distance_to_wall

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      time_step: 1.0e-3
      termination_step_count: 16
      time_stepping_type: adaptive
      time_step_count: 0

      realms: 
        - fluidRealm
        - ioRealm
