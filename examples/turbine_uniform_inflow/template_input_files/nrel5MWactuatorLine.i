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
    muelu_xml_file_name: ./milestone.xml
    summarize_muelu_timer: no

realms:

  - name: realm_1
    mesh: ../../mesh/nrel5MWactuatorLine.g
    use_edges: no
    automatic_decomposition_type: rcb

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
        target_name: Unspecified-2-HEX
        value:
          pressure: 0.0
          velocity: [8.0,0.0,0.0]

    material_properties:
      target_name: Unspecified-2-HEX
      specifications:
        - name: density
          type: constant
          value: 1.225

        - name: viscosity
          type: constant
          value: 1.846e-5

    boundary_conditions:

    - inflow_boundary_condition: bc_1
      target_name: west
      inflow_user_data:
        velocity: [8.0,0.0,0.0]

    - open_boundary_condition: bc_2
      target_name: east
      open_user_data:
        pressure: 0.0
        velocity: [8.0,0.0,0.0]

    - symmetry_boundary_condition: bc_3
      target_name: north
      symmetry_user_data:

    - symmetry_boundary_condition: bc_4
      target_name: south
      symmetry_user_data:

    - symmetry_boundary_condition: bc_5
      target_name: lower
      symmetry_user_data:

    - symmetry_boundary_condition: bc_6
      target_name: upper
      symmetry_user_data:

    solution_options:
      name: myOptions
      use_consolidated_solver_algorithm: yes

      options:

        - hybrid_factor:
            velocity: 1.0

        - limiter:
            pressure: no
            velocity: no

        - projected_nodal_gradient:
            pressure: element
            velocity: element

        - element_source_terms:
            momentum: [lumped_momentum_time_derivative, upw_advection_diffusion, lumped_actuator]
            continuity: [advection]

    actuator:
      type: ActLineFAST
      search_method: boost_rtree
      search_target_part: Unspecified-2-HEX

      n_turbines_glob: 1
      dry_run:  False
      debug:    True
      simStart: init
      t_start: 0.0
      t_max:    0.625
      dt_fast: 0.00625
      n_every_checkpoint: 100

      Turbine0:
        num_force_pts_blade: 50
        num_force_pts_tower: 20
        nacelle_cd: 1.0
        nacelle_area: 8.0
        air_density: 1.225
        epsilon: [ 5.0, 5.0, 5.0 ]
        turbine_base_pos: [ 0.0, 0.0, 0.0 ]
        turbine_hub_pos: [ 0.0, 0.0, 90.0 ]
        restart_filename: "blah"
        fast_input_filename: "nrel5mw.fst"
        turb_id:  1
        turbine_name: machine_one

    output:
      output_data_base_name: actuatorLine.e
      output_frequency: 1
      output_node_set: no
      output_variables:
       - velocity
       - pressure
       - actuator_source

    restart:
      restart_data_base_name: actuatorLine.rst
      restart_frequency: 10
      restart_start: 10
      compression_level: 9
      compression_shuffle: yes

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 10
      time_step: 0.0625
      time_stepping_type: fixed
      time_step_count: 0
      second_order_accuracy: no

      realms:
        - realm_1
