Simulations:
  - name: sim1
    time_integrator: ti_1
    optimizer: opt1

linear_solvers:

  - name: solve_mom
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
    max_iterations: 75
    kspace: 75
    output_level: 0
    recompute_preconditioner: false
    muelu_xml_file_name: ../../xml/milestone.xml

  - name: solve_other
    type: tpetra
    method: gmres
    preconditioner: sgs
    tolerance: 1e-5
    max_iterations: 50
    kspace: 50
    output_level: 0

realms:

  - name: realm_1
    mesh: heliumPlume.rst
    use_edges: no
    support_inconsistent_multi_state_restart: yes

    equation_systems:
      name: theEqSys
      max_iterations: 4    

      solver_system_specification:
        velocity: solve_mom
        turbulent_ke: solve_other
        mixture_fraction: solve_other
        pressure: solve_cont

      systems:
        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-2

        - TurbKineticEnergy:
            name: myTke
            max_iterations: 1
            convergence_tolerance: 1.e-2

        - MixtureFraction:
            name: myZ
            max_iterations: 1
            convergence_tolerance: 1.e-2

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
          type: mixture_fraction
          primary_value: 0.163e-3
          secondary_value: 1.18e-3

        - name: viscosity
          type: mixture_fraction
          primary_value: 1.967e-4
          secondary_value: 1.85e-4

    boundary_conditions:

    - inflow_boundary_condition: bc_inflow
      target_name: surface_1
      inflow_user_data:
        velocity: [0.0,34.0,0.0]
        turbulent_ke: 0.17
        mixture_fraction: 1.0

    - wall_boundary_condition: bc_bottom
      target_name: surface_2
      wall_user_data:
        velocity: [0,0,0]
        turbulent_ke: 0.0

    - open_boundary_condition: bc_side
      target_name: surface_3
      open_user_data:
        velocity: [0,0,0]
        pressure: 0.0
        turbulent_ke: 1.0e-16
        mixture_fraction: 0.0

    - open_boundary_condition: bc_top
      target_name: surface_4
      open_user_data:
        velocity: [0,0,0]
        pressure: 0.0
        turbulent_ke: 1.0e-16
        mixture_fraction: 0.0

    solution_options:
      name: myOptions
      turbulence_model: ksgs
      reduced_sens_cvfem_poisson: yes
      divU_stress_scaling: 1.0

      options:
        - hybrid_factor:
            velocity: 0.0
            turbulent_ke: 0.0
            mixture_fraction: 1.0

        - alpha_upw:
            velocity: 1.0
            turbulent_ke: 1.0
            mixture_fraction: 1.0

        - laminar_schmidt:
            turbulent_ke: 1.0
            mixture_fraction: 0.9

        - turbulent_schmidt:
            turbulent_ke: 1.0
            mixture_fraction: 1.0

        - source_terms:
            momentum: buoyancy
            continuity: density_time_derivative

        - element_source_terms:
            turbulent_ke: [ksgs_buoyant, NSO_4TH_ALT]

        - user_constants:
            gravity: [0.0,-981.0,0.0]
            reference_density: 1.18e-3

        - turbulence_model_constants:
             Cb2: 0.3501

    turbulence_averaging:
      time_filter_interval: 10.0
      specifications:
        - name: one
          target_name: block_1
          reynolds_averaged_variables:
            - velocity
            - mixture_fraction
            - turbulent_ke
          favre_averaged_variables:
            - velocity
            - mixture_fraction
            - turbulent_ke

          compute_reynolds_stress: yes
          compute_favre_stress: yes
          compute_favre_tke: yes
          compute_q_criterion: yes
          compute_vorticity: yes
          compute_lambda_ci: yes

    output:
      serialized_io_group_size: 2
      output_data_base_name: heliumPlume_rst.e
      output_frequency: 2
      output_node_set: no
      output_variables:
       - velocity
       - pressure
       - turbulent_ke
       - mixture_fraction
       - density
       - density_ra_one
       - turbulent_ke_ra_one
       - turbulent_ke_fa_one
       - mixture_fraction_ra_one
       - mixture_fraction_fa_one
       - velocity_ra_one
       - velocity_fa_one
       - reynolds_stress
       - favre_stress
       - resolved_favre_turbulent_ke
       - vorticity
       - q_criterion
       - lambda_ci

    restart:
      restart_data_base_name: heliumPlume_B.rst
      restart_frequency: 2 
      restart_time: 0.10

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_time: 0.30
      time_step: 1.0e-3
      time_stepping_type: adaptive
      time_step_count: 0
      second_order_accuracy: yes

      realms:
        - realm_1
