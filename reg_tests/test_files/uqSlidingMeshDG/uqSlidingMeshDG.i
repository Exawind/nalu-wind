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
    muelu_xml_file_name: ../../xml/milestone_aspect_ratio_smooth.xml
    recompute_preconditioner: no
    reuse_preconditioner: no

realms:

  - name: realm_1
    mesh: ../../mesh/uqvawt_corrected.exo
    use_edges: no
    activate_aura: no

    time_step_control:
     target_courant: 20.0
     time_step_change_factor: 1.15

    equation_systems:
      name: theEqSys
      max_iterations: 2

      solver_system_specification:
        pressure: solve_cont
        velocity: solve_scalar
        turbulent_ke: solve_scalar

      systems:
        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-2

        - TurbKineticEnergy:
            name: myTke
            max_iterations: 1
            convergence_tolerance: 1.e-2

    initial_conditions:

      - constant: ic_1
        target_name: [block_1, block_2, block_3, block_4, block_5, block_6, block_7]
        value:
          pressure: 0
          turbulent_ke: 1.0e-6

      - user_function: ic_2
        target_name: [block_1, block_2, block_3, block_4, block_5, block_6, block_7]
        user_function_name:
         velocity: wind_energy_taylor_vortex
        user_function_parameters:
         velocity: [-225.0,0.0,50.0,50.0,6.0]

    material_properties:
      target_name: [block_1, block_2, block_3, block_4, block_5, block_6, block_7]
      specifications:
        - name: density
          type: constant
          value: 1.226

        - name: viscosity
          type: constant
          value: 1.8e-5

    boundary_conditions:

    - inflow_boundary_condition: bc_left
      target_name: surface_1
      inflow_user_data:
        velocity: [6.0,0.0]
        turbulent_ke: 0.01

    - open_boundary_condition: bc_right
      target_name: surface_2
      open_user_data:
        pressure: 0.0
        turbulent_ke: 0.0

    - symmetry_boundary_condition: bc_top_bot
      target_name: surface_3
      symmetry_user_data:

    - wall_boundary_condition: bc_wingT
      target_name: surface_4
      wall_user_data:
        user_function_name:
         velocity: wind_energy
        user_function_string_parameters:
         velocity: [mmOne]
        use_wall_function: yes

    - wall_boundary_condition: bc_wingL
      target_name: surface_5
      wall_user_data:
        user_function_name:
         velocity: wind_energy
        user_function_string_parameters:
         velocity: [mmOne]
        use_wall_function: yes

    - wall_boundary_condition: bc_wingR
      target_name: surface_6
      wall_user_data:
        user_function_name:
         velocity: wind_energy
        user_function_string_parameters:
         velocity: [mmOne]
        use_wall_function: yes

    - non_conformal_boundary_condition: bc_top_out_in
      target_name: [surface_7, surface_8]
      non_conformal_user_data:
        expand_box_percentage: 5.0
        search_tolerance: 0.01

    - non_conformal_boundary_condition: bc_top_out_in
      target_name: [surface_8, surface_7]
      non_conformal_user_data:
        expand_box_percentage: 5.0
        search_tolerance: 0.01

    mesh_motion:
     - name: mmOne
       mesh_parts: [block_1, block_2, block_3, block_4, block_5]
       frame: non_inertial
       motion:
        - type: rotation
          omega: 1.0
          axis: [0.0,0.0,1.0]

     - name: mmTwo
       mesh_parts: [block_6, block_7]
       frame: non_inertial
       motion:
        - type: rotation
          omega: 0.0

    solution_options:
      name: myOptions
      turbulence_model: ksgs

      options:
        - hybrid_factor:
            velocity: 1.0
            turbulent_ke: 1.0

        - alpha_upw:
            velocity: 1.0
            turbulent_ke: 1.0

        - limiter:
            pressure: no
            velocity: no
            turbulent_ke: yes

        - projected_nodal_gradient:
            pressure: edge

        - non_conformal:
            gauss_labatto_quadrature: no
            upwind_advection: yes
            current_normal: yes

    output:
      output_data_base_name: uqSlidingMeshDG.e
      output_frequency: 10
      output_node_set: no
      output_variables:
       - dual_nodal_volume
       - velocity
       - pressure
       - turbulent_viscosity
       - turbulent_ke
       - mesh_displacement
       - pressure_force_ra_one
       - tau_wall_ra_one
       - yplus_ra_one
       - resolved_turbulent_ke_ra_one
       - reynolds_stress
       - element_reynolds
       - element_courant

    post_processing:

    - type: surface
      physics: surface_force_and_moment
      output_file_name: nalu_s4.dat
      frequency: 2
      parameters: [0,0]
      target_name: surface_4

    - type: surface
      physics: surface_force_and_moment
      output_file_name: nalu_s5.dat
      frequency: 2
      parameters: [0,0]
      target_name: surface_5

    - type: surface
      physics: surface_force_and_moment
      output_file_name: nalu_s6.dat
      frequency: 2
      parameters: [0,0]
      target_name: surface_6

    turbulence_averaging:
      time_filter_interval: 100000.0
      specifications:
        - name: one
          target_name: [surface_4, surface_5, surface_6]
          reynolds_averaged_variables:
            - pressure_force
            - tau_wall
            - yplus
            - resolved_turbulent_ke

          compute_tke: yes
          compute_reynolds_stress: yes

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_time: 0.04
      time_step: 0.002
      time_stepping_type: fixed
      time_step_count: 0

      realms:
        - realm_1
