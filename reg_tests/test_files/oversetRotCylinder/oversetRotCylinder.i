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

realms:

  - name: realm_1
    mesh: ../../mesh/oversetCylinder.g
    use_edges: no
    automatic_decomposition_type: rcb
    activate_aura: true

    equation_systems:
      name: theEqSys
      max_iterations: 3

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
          - Unspecified-2-HEX
          - Unspecified-3-HEX
        value:
          pressure: 0.0
          velocity: [1.0,0.0,0.0]

    material_properties:
      target_name:
          - Unspecified-2-HEX
          - Unspecified-3-HEX
      specifications:
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

    - symmetry_boundary_condition: bc_8
      target_name: side21
      symmetry_user_data:

    - symmetry_boundary_condition: bc_9
      target_name: side22
      symmetry_user_data:

    - overset_boundary_condition: bc_overset
      overset_connectivity_type: tioga
      overset_user_data:
        tioga_options:
          symmetry_direction: 2
          set_resolutions: no
        mesh_group:
          - overset_name: interior
            mesh_parts: [ Unspecified-2-HEX ]
            wall_parts: [ wall ]
            ovset_parts: [ overset1 ]

          - overset_name: wake
            mesh_parts: [ Unspecified-3-HEX]

    mesh_motion:
      - name: interior
        frame: non_inertial
        mesh_parts: [ Unspecified-2-HEX ]
        motion:
         - type: rotation
           omega: 6.00
           axis: [0.0, 1.0, 0.0]

    solution_options:
      name: myOptions

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

    post_processing:

      - type: surface
        physics: surface_force_and_moment
        output_file_name: oversetRotCylinder.dat
        frequency: 1
        parameters: [0,0]
        target_name: wall

    turbulence_averaging:
      time_filter_interval: 100000.0

      specifications:
        - name: one
          target_name: [Unspecified-2-HEX, Unspecified-3-HEX]
          compute_q_criterion: yes
          compute_vorticity: yes

    restart:
      restart_data_base_name: rst/oversetRotCylinder.rst
      restart_frequency: 100
      restart_start: 100

    output:
      output_data_base_name: out/oversetRotCylinder.exo
      output_frequency: 1000
      output_node_set: no
      output_variables:
       - velocity
       - pressure
       - iblank
       - iblank_cell
       - mesh_displacement
       - q_criterion
       - vorticity



Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 50
      time_step: 0.0001
      time_stepping_type: adaptive
      time_step_count: 0
      second_order_accuracy: yes

      realms:
        - realm_1
