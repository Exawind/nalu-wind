# -*- mode: yaml -*-

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
    max_iterations: 200
    kspace: 50
    output_level: 0

  - name: solve_cont
    type: hypre
    method: hypre_gmres
    preconditioner: boomerAMG
    tolerance: 1e-5
    max_iterations: 200
    kspace: 20
    output_level: 0

realms:

  - name: background
    mesh: ../../mesh/oversetCylinder.g
    use_edges: yes
    automatic_decomposition_type: rcb

    equation_systems: &eqsys1
      name: theEqSys
      max_iterations: 1
      decoupled_overset_solve: yes

      solver_system_specification:
        velocity: solve_scalar
        pressure: solve_cont

      systems:

        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-7

    initial_conditions:

      - constant: ic_1
        target_name:
          - Unspecified-3-HEX
        value:
          pressure: 0.0
          velocity: [1.0,0.0,0.0]

    material_properties:
      target_name:
        - Unspecified-3-HEX
      specifications: &matpropspec1
        - name: density
          type: constant
          value: 1.00

        - name: viscosity
          type: constant
          value: 0.005

    boundary_conditions:

    - inflow_boundary_condition: bc_1
      target_name: inlet
      inflow_user_data:
        velocity: [1.0,0.0,0.0]
        pressure: 0.0

    - open_boundary_condition: bc_2
      target_name: outlet
      open_user_data:
        pressure: 0.0
        velocity: [0.0,0.0,0.0]

    - symmetry_boundary_condition: bc_3
      target_name: top
      symmetry_user_data:

    - symmetry_boundary_condition: bc_4
      target_name: bottom
      symmetry_user_data:

    - symmetry_boundary_condition: bc_8
      target_name: side21
      symmetry_user_data:

    - symmetry_boundary_condition: bc_9
      target_name: side22
      symmetry_user_data:

    - overset_boundary_condition: bc_overset
      overset_connectivity_type: tioga
      overset_user_data:
        tioga_options: &tiogaopts1
          symmetry_direction: 2
          set_resolutions: yes
        mesh_group:
          - overset_name: background
            mesh_parts: [ Unspecified-3-HEX]

    solution_options: &solnopts1
      name: myOptions
      projected_timescale_type: momentum_diag_inv #### Use 1/diagA formulation

      options:
        - hybrid_factor:
            velocity: 1.0

        - upw_factor:
            velocity: 1.0

        - alpha_upw:
            velocity: 1.0

        - limiter:
            pressure: no
            velocity: no

        - projected_nodal_gradient:
            pressure: element
            velocity: element

        - relaxation_factor:
            velocity: 0.7
            pressure: 0.3
            turbulent_ke: 0.7
            specific_dissipation_rate: 0.7

    output:
      output_data_base_name: out/back.e
      output_frequency: 10
      output_node_set: no
      output_variables:
       - velocity
       - pressure
       - dpdx
       - mesh_displacement
       - iblank
       - iblank_cell

  - name: cylinder
    mesh: ../../mesh/oversetCylinder.g
    use_edges: yes
    automatic_decomposition_type: rcb

    equation_systems: *eqsys1

    initial_conditions:
      - constant: ic_1
        target_name:
          - Unspecified-2-HEX
        value:
          pressure: 0.0
          velocity: [1.0,0.0,0.0]

    material_properties:
      target_name:
          - Unspecified-2-HEX
      specifications: *matpropspec1

    boundary_conditions:
    - wall_boundary_condition: bc_5
      target_name: wall
      wall_user_data:
        user_function_name:
         velocity: wind_energy
        user_function_string_parameters:
         velocity: [interior]

    - symmetry_boundary_condition: bc_6
      target_name: side11
      symmetry_user_data:

    - symmetry_boundary_condition: bc_7
      target_name: side12
      symmetry_user_data:

    - overset_boundary_condition: bc_overset
      overset_connectivity_type: tioga
      overset_user_data:
        # This is important to ensure unique mesh tags for TIOGA
        mesh_tag_offset: 1
        tioga_options: *tiogaopts1
        mesh_group:
          - overset_name: cylinder
            mesh_parts: [ Unspecified-2-HEX ]
            wall_parts: [ wall ]
            ovset_parts: [ overset1 ]

    mesh_motion:
      - name: interior
        mesh_parts: [ Unspecified-2-HEX ]
        motion:
         - type: rotation
           omega: 6.00
           axis: [0.0, 1.0, 0.0]

    solution_options: *solnopts1

    output:
      output_data_base_name: out/cyl.e
      output_frequency: 10
      output_node_set: no
      output_variables:
       - velocity
       - pressure
       - dpdx
       - mesh_displacement
       - iblank
       - iblank_cell


Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 5
      time_step: 0.003
      time_stepping_type: fixed
      time_step_count: 0
      second_order_accuracy: yes
      nonlinear_iterations: 4

      realms:
        - background
        - cylinder
