Simulations:
  - name: sim1
    time_integrator: ti_1
    optimizer: opt1

linear_solvers:

  - name: solve_scalar
    type: tpetra
    method: tfqmr 
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
    max_iterations: 100 
    kspace: 100
    output_level: 0
    muelu_xml_file_name: ../../xml/milestone_aspect_ratio.xml

realms:

  - name: realm_1
    mesh: ../../mesh/hybrid.g
    use_edges: no 

    time_step_control:
     target_courant: 2.0
     time_step_change_factor: 1.2
   
    equation_systems:
      name: theEqSys
      max_iterations: 2 

      solver_system_specification:
        velocity: solve_scalar
        mixture_fraction: solve_scalar
        pressure: solve_cont

      systems:
        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-5

        - MixtureFraction:
            name: myZ
            max_iterations: 1
            convergence_tolerance: 1.e-5

    initial_conditions:
      - constant: ic_1
        target_name: [block_1, block_2, block_3, block_4, block_5, block_6]
        value:
          pressure: 0
          velocity: [11.5,0,0]
          mixture_fraction: 0.0

    material_properties:
      target_name: [block_1, block_2, block_3, block_4, block_5, block_6]
      specifications:
        - name: density
          type: constant
          value: 1.1814
        - name: viscosity
          type: constant
          value: 1.79e-5

    boundary_conditions:

    - inflow_boundary_condition: bc_inflowT
      target_name: surface_1
      inflow_user_data:
        velocity: [11.5,0,0]
        mixture_fraction: 1.0

    - inflow_boundary_condition: bc_inflowB
      target_name: surface_2
      inflow_user_data:
        velocity: [11.5,0,0]
        mixture_fraction: 1.0

    - open_boundary_condition: bc_back
      target_name: surface_3
      open_user_data:
        velocity: [0,0,0]
        pressure: 0.0
        mixture_fraction: 0.0

    - symmetry_boundary_condition: bc_sides
      target_name: surface_4
      symmetry_user_data:

    - symmetry_boundary_condition: bc_top
      target_name: surface_5
      symmetry_user_data:

    - symmetry_boundary_condition: bc_bot
      target_name: surface_6
      symmetry_user_data:

    solution_options:
      name: myOptions
      turbulence_model: smagorinsky  

      use_consolidated_solver_algorithm: yes
      use_consolidated_face_elem_bc_algorithm: yes

      options:
        - hybrid_factor:
            mixture_fraction: 1.0

        - alpha_upw:
            mixture_fraction: 1.0

        - laminar_schmidt:
            mixture_fraction: 0.5

        - turbulent_schmidt:
            mixture_fraction: 1.0

        - element_source_terms:
            momentum: [lumped_momentum_time_derivative, advection_diffusion]
            continuity: [advection]
            mixture_fraction: [lumped_mixture_fraction_time_derivative, upw_advection_diffusion]
          
    output:
      output_data_base_name: output.e
      output_frequency: 5
      output_node_set: no 
      output_variables:
       - velocity
       - pressure
       - mixture_fraction
       - scalar_variance
       - scalar_dissipation
       - turbulent_viscosity
       - element_reynolds
       - element_courant

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_time: 0.1
      time_step: 0.01
      time_stepping_type: adaptive 
      time_step_count: 0
      second_order_time_accuracy: yes

      realms: 
        - realm_1
