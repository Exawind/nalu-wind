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
    muelu_xml_file_name: ../../xml/milestone-nc.xml

realms:

  - name: realm_1
    mesh: ../../mesh/nonconformal_2blk_2D.g
    use_edges: no

    equation_systems:
      name: theEqSys
      max_iterations: 2

      solver_system_specification:
        pressure: solve_cont
        velocity: solve_scalar

      systems:
        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-5

    initial_conditions:
      - constant: ic_1
        target_name: [block_2, block_3]
        value:
          pressure: 0
          velocity: [100.0,0.0]

    material_properties:
      target_name: [block_2, block_3]
      specifications:
        - name: density
          type: constant
          value: 1.0e-3

        - name: viscosity
          type: constant
          value: 1.8e-4

    boundary_conditions:

    - periodic_boundary_condition: bc_front
      target_name: [surface_3, surface_4]
      periodic_user_data:
        search_tolerance: 0.0001

    - inflow_boundary_condition: bc_front
      target_name: surface_1
      inflow_user_data:
        velocity: [100.0,0.0,0.0]

    - open_boundary_condition: bc_back
      target_name: surface_2
      open_user_data:
        pressure: 0.0

    - non_conformal_boundary_condition: bc_in_out
      target_name: [surface_5, surface_6]
      non_conformal_user_data:
        expand_box_percentage: 5.0

    - non_conformal_boundary_condition: bc_out_in
      target_name: [surface_6, surface_5]
      non_conformal_user_data:
        expand_box_percentage: 5.0

    solution_options:
      name: myOptions

      use_consolidated_solver_algorithm: yes

      options:
        - hybrid_factor:
            velocity: 1.0

        - alpha_upw:
            velocity: 1.0

        - non_conformal:
            gauss_labatto_quadrature: no
            upwind_advection: yes

        - element_source_terms:
            momentum: [lumped_momentum_time_derivative, upw_advection_diffusion]
            continuity: [advection]

    output:
      output_data_base_name: nonConformalWithPeriodicConsolidated.e
      output_frequency: 5
      output_node_set: no
      output_variables:
       - velocity
       - pressure

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 100
      time_step: 0.0010
      time_stepping_type: adaptive
      time_step_count: 0

      realms:
        - realm_1
