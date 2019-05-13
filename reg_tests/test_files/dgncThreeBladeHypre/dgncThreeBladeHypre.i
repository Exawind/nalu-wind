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
    type: hypre
    method: hypre_gmres
    preconditioner: boomerAMG
    tolerance: 1e-5
    max_iterations: 200
    kspace: 5
    output_level: 0
    recompute_preconditioner: false

  - name: solve_momentum
    type: hypre
    method: hypre_gmres
    preconditioner: boomerAMG
    tolerance: 1e-5
    max_iterations: 200
    kspace: 5
    output_level: 0
    recompute_preconditioner: true
    segregated_solver: yes
    absolute_tolerance: 1.0e-12
    bamg_max_levels: 1
    bamg_num_sweeps: 1
    bamg_relax_type: 6

realms:

  - name: realm_1
    mesh: ../../mesh/threeBladeMesh.g # MKS
    use_edges: no
    activate_aura: no
    automatic_decomposition_type: rcb

    time_step_control:
     target_courant: 2.0
     time_step_change_factor: 1.15

    equation_systems:
      name: theEqSys
      max_iterations: 2

      solver_system_specification:
        pressure: solve_cont
        velocity: solve_momentum
        mixture_fraction: solve_scalar
        dpdx: solve_scalar
        duidx: solve_scalar
        dzdx: solve_scalar

      systems:
        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-5

        - MixtureFraction:
            name: myZ
            max_iterations: 1
            convergence_tolerance: 1.e-2
            output_clipping_diagnostic: yes

    initial_conditions:
      - constant: ic_1
        target_name: [block_1, block_2, block_3, block_4]
        value:
          pressure: 0
          velocity: [0.5,0.0]
          mixture_fraction: 0.0

    material_properties:
      target_name: [block_1, block_2, block_3, block_4]
      specifications:
        - name: density
          type: constant
          value: 1.0

        - name: viscosity
          type: constant
          value: 1.8e-5

    boundary_conditions:

    - inflow_boundary_condition: bc_left
      target_name: surface_1
      inflow_user_data:
        velocity: [0.5,0.0,0.0]
        mixture_fraction: 0.0

    - open_boundary_condition: bc_right
      target_name: surface_2
      open_user_data:
        pressure: 0.0
        mixture_fraction: 0.0

    - symmetry_boundary_condition: bc_top
      target_name: surface_3
      symmetry_user_data:

    - symmetry_boundary_condition: bc_bot
      target_name: surface_4
      symmetry_user_data:

    - wall_boundary_condition: bc_front_blade
      target_name: surface_5
      wall_user_data:
        user_function_name:
         velocity: wind_energy
        user_function_string_parameters:
         velocity: [mmFront_ss5]
        mixture_fraction: 1.0

    - wall_boundary_condition: bc_back_lower_blade
      target_name: surface_6
      wall_user_data:
        user_function_name:
         velocity: wind_energy
        user_function_string_parameters:
         velocity: [mmBot_ss6]
        mixture_fraction: 1.0

    - wall_boundary_condition: bc_back_higher_blade
      target_name: surface_7
      wall_user_data:
        user_function_name:
         velocity: wind_energy
        user_function_string_parameters:
         velocity: [mmTop_ss7]
        mixture_fraction: 1.0

    - non_conformal_boundary_condition: bc_in_out
      current_target_name: [surface_8, surface_9, surface_10]
      opposing_target_name: [surface_88, surface_99, surface_1000]
      non_conformal_user_data:
        expand_box_percentage: 5.0
        search_tolerance: 0.01

    - non_conformal_boundary_condition: bc_out_in
      current_target_name: [surface_88, surface_99, surface_1000]
      opposing_target_name: [surface_8, surface_9, surface_10]
      non_conformal_user_data:
        expand_box_percentage: 5.0
        search_tolerance: 0.01

    mesh_motion:

      - name: mmFront_ss5
        mesh_parts: [block_2]
        frame: non_inertial
        compute_centroid: yes
        motion:
         - type: rotation
           omega: 3.14
           axis: [0.0,0.0,1.0]

      - name: mmTop_ss7
        mesh_parts: [block_3]
        frame: non_inertial
        compute_centroid: yes
        motion:
         - type: rotation
           omega: 6.28
           axis: [0.0,0.0,1.0]

      - name: mmBot_ss6
        mesh_parts: [block_4]
        frame: non_inertial
        compute_centroid: yes
        motion:
         - type: rotation
           omega: 1.57
           axis: [0.0,0.0,-1.0]

    solution_options:
      name: myOptions

      use_consolidated_solver_algorithm: yes

      options:

        - limiter:
            pressure: no
            velocity: no

        - hybrid_factor:
            mixture_fraction: 0.0
            velocity: 0.0

        - element_source_terms:
            momentum: [lumped_momentum_time_derivative, advection_diffusion, NSO_2ND_KE]
            continuity: advection
            mixture_fraction: [lumped_mixture_fraction_time_derivative, advection_diffusion, NSO_2ND_ALT]

        - non_conformal:
            gauss_labatto_quadrature: no
            upwind_advection: yes
            current_normal: yes
            include_png_penalty: yes

        - consistent_mass_matrix_png:
            pressure: yes
            velocity: yes
            mixture_fraction: yes

    output:
      output_data_base_name: dgNonConformalThreeBlade.e
      output_frequency: 20
      output_node_set: no
      output_variables:
       - velocity
       - pressure
       - mixture_fraction
       - mesh_displacement

    restart:
      restart_data_base_name: dgNonConformalThreeBlade.rst
      restart_frequency: 25
      restart_start: 25

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 10
      time_step: 0.001
      time_stepping_type: adaptive
      time_step_count: 0
      second_order_accuracy: yes

      realms:
        - realm_1
