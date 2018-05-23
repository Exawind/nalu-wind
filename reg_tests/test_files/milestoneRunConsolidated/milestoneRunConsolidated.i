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
    max_iterations: 75 
    kspace: 75 
    output_level: 0

  - name: solve_cont
    type: tpetra
    method: gmres
    preconditioner: muelu
    tolerance: 1e-5
    max_iterations: 75
    kspace: 75
    muelu_xml_file_name: ../../xml/milestone.xml
    output_level: 0
    recompute_preconditioner: no

realms:

  - name: realm_1
    mesh: ../../mesh/1cm_ped_35K.g
    use_edges: no
    automatic_decomposition_type: rcb
   
    equation_systems:
      name: theEqSys
      max_iterations: 2 

      solver_system_specification:
        velocity: solve_scalar
        turbulent_ke: solve_scalar
        mixture_fraction: solve_scalar
        pressure: solve_cont

      systems:
        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-5

        - TurbKineticEnergy:
            name: myTke
            max_iterations: 1
            convergence_tolerance: 1.e-5

        - MixtureFraction:
            name: myZ
            max_iterations: 1
            convergence_tolerance: 1.e-5

    initial_conditions:
      - constant: ic_1
        target_name: block_1
        value:
          pressure: 0
          velocity: [0,0,0]
          turbulent_ke: 0.0
          mixture_fraction: 0.0

    material_properties:
      target_name: block_1
      specifications:
        - name: density
          type: constant
          value: 1.1814e-3
        - name: viscosity
          type: constant
          value: 1.79e-4

    boundary_conditions:

    - wall_boundary_condition: bc_bottom
      target_name: surface_1
      wall_user_data:
        velocity: [0,0,0]
        turbulent_ke: 0.0

    - inflow_boundary_condition: bc_inflow
      target_name: surface_2
      inflow_user_data:
        velocity: [0,0,1000.0]
        turbulent_ke: 150.0 
        mixture_fraction: 1.0

    - wall_boundary_condition: bc_pipe
      target_name: surface_3
      wall_user_data:
        velocity: [0,0,0]
        turbulent_ke: 0.0

    - open_boundary_condition: bc_side
      target_name: surface_4
      open_user_data:
        velocity: [0,0,0]
        pressure: 0.0
        turbulent_ke: 1.0e-16
        mixture_fraction: 0.0

    - open_boundary_condition: bc_top
      target_name: surface_5
      open_user_data:
        velocity: [0,0,0]
        pressure: 0.0
        turbulent_ke: 1.0e-16
        mixture_fraction: 0.0

    solution_options:
      name: myOptions
      turbulence_model: ksgs

      use_consolidated_solver_algorithm: yes
      use_consolidated_face_elem_bc_algorithm: yes

      options:
        - hybrid_factor:
            velocity: 0.0 
            turbulent_ke: 1.0
            mixture_fraction: 1.0

        - alpha_upw:
            velocity: 1.0 
            turbulent_ke: 1.0
            mixture_fraction: 1.0

        - laminar_schmidt:
            turbulent_ke: 0.9 
            mixture_fraction: 1.0 

        - turbulent_schmidt:
            turbulent_ke: 1.0
            mixture_fraction: 1.0

        - limiter:
            velocity: no
            turbulent_ke: yes
            mixture_fraction: yes
                
        - element_source_terms:
            momentum: [lumped_momentum_time_derivative, advection_diffusion]
            continuity: [advection]
            mixture_fraction: [lumped_mixture_fraction_time_derivative, upw_advection_diffusion]
            turbulent_ke: [lumped_turbulent_ke_time_derivative, upw_advection_diffusion, ksgs]

    output:
      output_data_base_name: milestoneRunConsolidated.e
      output_frequency: 10
      output_node_set: no
      output_variables:
       - velocity
       - pressure
       - turbulent_ke
       - mixture_fraction
       - scalar_variance
       - scalar_dissipation

    restart:
      restart_data_base_name: milestoneRunConsolidated_A.rst
      restart_frequency: 25
    
Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 50 
      time_step: 1.0e-4
      time_stepping_type: fixed 
      time_step_count: 0
      second_order_accuracy: no

      realms: 
        - realm_1
