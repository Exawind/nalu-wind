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

realms:

  - name: realm_1
    mesh: ../../mesh/NACA.g
    use_edges: yes
    check_for_missing_bcs: no

    equation_systems:
      name: theEqSys
      max_iterations: 2

      solver_system_specification:
        pressure: solve_cont
        velocity: solve_scalar
        mixture_fraction: solve_scalar

      systems:
        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-5

        - MixtureFraction:
            name: myZ
            max_iterations: 1
            convergence_tolerance: 1.e-2

    initial_conditions:

      - constant: ic_1
        target_name: [block_1, block_2]
        value:
          pressure: 0
          mixture_fraction: 0.0
          velocity: [0.0,0.0]

    material_properties:
      target_name: [block_1, block_2]
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
        velocity: [10.0,0.0]
        mixture_fraction: 1.0

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

    - non_conformal_boundary_condition: bc_in_out
      target_name: [surface_5, surface_6]
      non_conformal_user_data:
        expand_box_percentage: 50.0

    - non_conformal_boundary_condition: bc_out_in
      target_name: [surface_6, surface_5]
      non_conformal_user_data:
        expand_box_percentage: 50.0

    - wall_boundary_condition: bc_wing
      target_name: surface_7
      wall_user_data:
        velocity: [0.0,0.0]
        mixture_fraction: 0.0

    mesh_motion:
    - name: mmOne
      mesh_parts: [block_1]
      frame: non_inertial
      motion:
       - type: rotation
         omega: 1.0

    - name: mmTwo
      mesh_parts: [block_2]
      frame: non_inertial
      motion:
       - type: rotation
         omega: 0.0

    solution_options:
      name: myOptions

      options:
        - hybrid_factor:
            velocity: 1.0
            mixture_fraction: 1.0

        - non_conformal:
            gauss_labatto_quadrature: no

    output:
      output_data_base_name: output.e
      output_frequency: 10
      output_node_set: no
      output_variables:
       - dual_nodal_volume
       - velocity
       - pressure
       - mixture_fraction
       - mesh_displacement

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 100
      time_step: 0.0050
      time_stepping_type: fixed
      time_step_count: 0

      realms:
        - realm_1
