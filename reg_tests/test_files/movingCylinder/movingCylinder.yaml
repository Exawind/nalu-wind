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
    muelu_xml_file_name: ../../xml/milestone_aspect_ratio.xml

transfers:

# Fluid to ....
          
# MM to ....

  - name: xfer_mm_fluid_surf
    type: geometric
    realm_pair: [mmRealm, fluidRealm]
    mesh_part_pair: [surface_5, surface_5]
    transfer_variables:
      - [mesh_velocity, velocity_bc]

  - name: xfer_mm_fluid_vol
    type: geometric
    realm_pair: [mmRealm, fluidRealm]
    mesh_part_pair: [block_10, block_10]
    transfer_variables:
      - [mesh_velocity, mesh_velocity]
      - [mesh_displacement, mesh_displacement]
      - [div_mesh_velocity, div_mesh_velocity]

realms:

  - name: mmRealm
    mesh: ../../mesh/waterChannel_cgs.g
    use_edges: no 
    check_jacobians: yes

    equation_systems:
      name: theEqSys
      max_iterations: 1 
    
      solver_system_specification:
        mesh_displacement: solve_scalar

      systems:
        - MeshDisplacement:
            name: myMM
            max_iterations: 1 
            convergence_tolerance: 1.e-3
            activate_mass: yes
            deform_wrt_model_coordinates: yes

    boundary_conditions:

    - wall_boundary_condition: bc_left
      target_name: surface_1
      wall_user_data:
        mesh_displacement: [0.0,0.0]

    - wall_boundary_condition: bc_right
      target_name: surface_2
      wall_user_data:
        mesh_displacement: [0.0,0.0]

    - wall_boundary_condition: bc_bottom
      target_name: surface_3
      wall_user_data:
        mesh_displacement: [0.0,0.0]

    - wall_boundary_condition: bc_top
      target_name: surface_4
      wall_user_data:
        mesh_displacement: [0.0,0.0]

    - wall_boundary_condition: bc_cyliner
      target_name: surface_5
      wall_user_data:
        user_function_name:
         mesh_displacement: sinusoidal
        user_function_parameters:
         mesh_displacement: [0.20]


    solution_options:
      name: myOptions

      options:
        - projected_nodal_gradient:
            mesh_velocity: element
    
    initial_conditions:

      - constant: ic_1
        target_name: block_10
        value:
          mesh_displacement: [0.0,0.0]

    material_properties:
      target_name: block_10
      specifications:
        - name: lame_mu
          type: geometric
        - name: lame_lambda
          type: geometric
        - name: density
          type: constant
          value: 10.0

    output:
      output_data_base_name: meshDisplacement.e
      output_frequency: 20
      output_node_set: no
      output_variables:
       - mesh_displacement
       - mesh_displacement_bc
       - dxTmp
       - mesh_velocity
       - dvdx

  - name: fluidRealm
    mesh: ../../mesh/waterChannel_cgs.g
    use_edges: no 

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
      target_name: block_10
      specifications:
        - name: density
          type: constant
          value: 1.0

        - name: viscosity
          type: constant
          value: 0.0075

    initial_conditions:
      - constant: ic_1
        target_name: block_10
        value:
          pressure: 0.0
          velocity: [0.0,0.0]
  
    boundary_conditions:

    - inflow_boundary_condition: bc_1
      target_name: surface_1
      inflow_user_data:
        velocity: [10.0,0.0]

    - open_boundary_condition: bc_2
      target_name: surface_2
      open_user_data:
        pressure: 0.0
        velocity: [0.0,0.0]

    - wall_boundary_condition: bc_3
      target_name: surface_3
      wall_user_data:
         velocity: [0.0,0.0]

    - symmetry_boundary_condition: bc_4
      target_name: surface_4
      symmetry_user_data:

    - wall_boundary_condition: bc_5
      target_name: surface_5
      wall_user_data:
         velocity: [0.0,0.0]
         fsi_interface: yes

    solution_options:
      name: myOptions
      externally_provided_mesh_deformation: yes

      options:
        - hybrid_factor:
            velocity: 1.0

        - limiter:
            pressure: no
            velocity: no

        - source_terms:
            momentum: gcl
            continuity: gcl

        - projected_nodal_gradient:
            momentum: element
            continuity: element
    
    output:
      output_data_base_name: fluid.e
      output_frequency: 20
      output_node_set: no
      output_variables:
       - velocity
       - mesh_displacement
       - pressure
       - mesh_velocity
       - div_mesh_velocity

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 20
      time_step: 0.004
      time_stepping_type: fixed
      time_step_count: 0
      nonlinear_iterations: 1

      realms:
        - mmRealm
        - fluidRealm

      transfers:
        - xfer_mm_fluid_surf
        - xfer_mm_fluid_vol
