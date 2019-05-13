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
    summarize_muelu_timer: no

realms:

  - name: realm_1
    mesh: ../../mesh/oversetMeshAligned.g
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
        target_name: [block_1, block_2]
        value:
          pressure: 0.0
          velocity: [0.0,0.0]

    material_properties:
      target_name: [block_1, block_2]
      specifications:
        - name: density
          type: constant
          value: 1.18

        - name: viscosity
          type: constant
          value: 1.8e-5

    boundary_conditions:

    - inflow_boundary_condition: bc_1
      target_name: surface_1
      inflow_user_data:
        velocity: [0.50,0.0]

    - open_boundary_condition: bc_2
      target_name: surface_2
      open_user_data:
        pressure: 0.0
        velocity: [0.0,0.0,0.0]

    - symmetry_boundary_condition: bc_3
      target_name: surface_3
      symmetry_user_data:

    - symmetry_boundary_condition: bc_4
      target_name: surface_4
      symmetry_user_data:

    - wall_boundary_condition: bc_5
      target_name: surface_5
      wall_user_data:
        velocity: [0.0,0.0]

    - overset_boundary_condition: bc_left
      overset_user_data:
        percent_overlap: 10.0
        background_block: block_1
        overset_block: block_2
        overset_surface: surface_6
        background_cut_block: block_3
        background_cut_surface: surface_101

    solution_options:
      name: myOptions

      options:

        - hybrid_factor:
            velocity: 1.0

        - limiter:
            pressure: no
            velocity: no

        - projected_nodal_gradient:
            pressure: element
            velocity: element

    output:
      output_data_base_name: oversetFluids.e
      output_frequency: 20
      output_node_set: no
      output_variables:
       - dual_nodal_volume
       - velocity
       - pressure
       - dpdx
       - dudx
       - intersected_element

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 100
      time_step: 0.001
      time_stepping_type: adaptive
      time_step_count: 0
      second_order_accuracy: no

      realms:
        - realm_1
