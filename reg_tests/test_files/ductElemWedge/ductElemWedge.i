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
    muelu_xml_file_name: ../../xml/matches_ml_default.xml

realms:

  - name: fluidRealm
    mesh: ../../mesh/ductwedge.g
    use_edges: no

    time_step_control:
     target_courant: 2.0
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

    initial_conditions:
      - constant: ic_1
        target_name: block_1
        value:
          pressure: 0
          velocity: [0,0,0]

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

    - inflow_boundary_condition: bc_inflow
      target_name: surface_1
      inflow_user_data:
        velocity: [0.0,0.0,2.0]

    - open_boundary_condition: bc_open
      target_name: surface_2
      open_user_data:
        velocity: [0,0,0]
        pressure: 0.0

    - wall_boundary_condition: bc_wall
      target_name: surface_3
      wall_user_data:
        velocity: [0,0,0]
        use_wall_function: yes

    solution_options:
      name: myOptions
      turbulence_model: wale

      use_consolidated_solver_algorithm: yes
      use_consolidated_face_elem_bc_algorithm: yes

      options:
      
        - element_source_terms:
            momentum: [lumped_momentum_time_derivative, advection_diffusion]
            continuity: advection

        - noc_correction:
            pressure: yes

        - projected_nodal_gradient:
            pressure: element
          
    output:
      output_data_base_name: ductElemWedge.e
      output_frequency: 2 
      output_node_set: no 
      output_variables:
       - velocity
       - pressure

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      time_step: 0.1
      termination_time: 1.5
      time_stepping_type: adaptive
      time_step_count: 0
      second_order_accuracy: yes

      realms: 
        - fluidRealm
