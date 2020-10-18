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
    max_iterations: 50
    kspace: 50
    output_level: 0

  - name: solve_cont
    type: tpetra
    method: gmres 
    preconditioner: muelu 
    tolerance: 1e-3
    max_iterations: 75
    kspace: 75
    output_level: 0
    muelu_xml_file_name: ../../xml/milestone_aspect_ratio.xml
    summarize_muelu_timer: no

realms:

  - name: realm_1
    mesh: ../../mesh/oversetSphereTioga.g
    use_edges: no
    automatic_decomposition_type: rcb
    check_for_missing_bcs: yes

    time_step_control:
      target_courant: 2.0
      time_step_change_factor: 1.1

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
        target_name:
          - Unspecified-2-HEX
          - Unspecified-3-HEX
          - Unspecified-4-HEX
          - Unspecified-5-HEX
          - Unspecified-6-HEX
          - Unspecified-7-HEX
          - Unspecified-8-HEX
        value:
          pressure: 0.0
          velocity: [1.0,0.0,0.0]

    material_properties:
      target_name:
          - Unspecified-2-HEX
          - Unspecified-3-HEX
          - Unspecified-4-HEX
          - Unspecified-5-HEX
          - Unspecified-6-HEX
          - Unspecified-7-HEX
          - Unspecified-8-HEX

      specifications:
        - name: density
          type: constant
          value: 1.00

        - name: viscosity
          type: constant
          value: 0.001

    boundary_conditions:

    - inflow_boundary_condition: bc_1
      target_name: inlet
      inflow_user_data:
        velocity: [1.0,0.0,0.0]

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

    - symmetry_boundary_condition: bc_6
      target_name: left
      symmetry_user_data:

    - symmetry_boundary_condition: bc_7
      target_name: right
      symmetry_user_data:

    - wall_boundary_condition: bc_5
      target_name: wall
      wall_user_data:
        velocity: [0.0,0.0,0.0]

    - overset_boundary_condition: bc_overset
      overset_connectivity_type: tioga
      overset_user_data:
        tioga_options:
          set_resolutions: no
        mesh_group:
          - overset_name: interior
            mesh_parts:
              - Unspecified-2-HEX
              - Unspecified-3-HEX
              - Unspecified-4-HEX
              - Unspecified-5-HEX
              - Unspecified-6-HEX
              - Unspecified-7-HEX
            wall_parts: [ wall ]
            ovset_parts: [ overset ]

          - overset_name: exterior
            mesh_parts: [ Unspecified-8-HEX ]


    solution_options:
      name: myOptions
      turbulence_model: laminar

      reduced_sens_cvfem_poisson: yes

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
      output_data_base_name: out/sphere.e
      output_frequency: 50
      output_node_set: no 
      output_variables:
       - velocity
       - pressure
       - iblank
       - iblank_cell

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 10
      time_step: 0.025
      time_stepping_type: adaptive
      time_step_count: 0
      second_order_accuracy: no

      realms:
        - realm_1
