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
    method: gmres
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
    tolerance: 1e-4
    max_iterations: 50
    kspace: 50
    output_level: 0

transfers:

# Fluid to ....

  - name: xfer_fluid_io
    type: geometric
    realm_pair: [fluidRealm, ioRealm]
    mesh_part_pair: [block_1, block_1]
    objective: input_output
    transfer_variables:
      - [mixture_fraction, mixture_fraction]
      - [reynolds_stress, reynolds_stress]

realms:

  - name: fluidRealm
    mesh: ../../mesh/100cm_13K_S_R1.g
    use_edges: yes 

    equation_systems:
      name: theEqSys
      max_iterations: 4    

      solver_system_specification:
        velocity: solve_mom
        mixture_fraction: solve_other
        pressure: solve_cont

      systems:
        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-2

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
        mixture_fraction: 1.0

    - wall_boundary_condition: bc_bottom
      target_name: surface_2
      wall_user_data:
        velocity: [0,0,0]

    - open_boundary_condition: bc_side
      target_name: surface_3
      open_user_data:
        velocity: [0,0,0]
        pressure: 0.0
        mixture_fraction: 0.0

    - open_boundary_condition: bc_top
      target_name: surface_4
      open_user_data:
        velocity: [0,0,0]
        pressure: 0.0
        mixture_fraction: 0.0

    solution_options:
      name: myOptions
      turbulence_model: smagorinsky

      divU_stress_scaling: 1.0

      options:
        - hybrid_factor:
            velocity: 0.0
            mixture_fraction: 1.0

        - alpha_upw:
            velocity: 1.0
            mixture_fraction: 1.0

        - laminar_schmidt:
            mixture_fraction: 0.9

        - turbulent_schmidt:
            mixture_fraction: 1.0

        - source_terms:
            momentum: buoyancy
#            continuity: density_time_derivative

        - user_constants:
            gravity: [0.0,-981.0,0.0]
            reference_density: 1.18e-3

    turbulence_averaging:
      time_filter_interval: 10.0
      specifications:
        - name: one
          target_name: block_1
          reynolds_averaged_variables:
            - velocity
            - mixture_fraction
            - resolved_turbulent_ke
          favre_averaged_variables:
            - velocity
            - mixture_fraction
            - resolved_turbulent_ke

          compute_tke: yes 
          compute_reynolds_stress: yes
          compute_mean_resolved_ke: yes

    output:
      serialized_io_group_size: 2
      output_data_base_name: heliumPlume.e
      output_frequency: 4 
      output_node_set: no
      compression_level: 9
      compression_shuffle: yes
      output_variables:
       - velocity
       - pressure
       - mixture_fraction
       - density
       - density_ra_one
       - mixture_fraction_ra_one
       - mixture_fraction_fa_one
       - velocity_ra_one
       - velocity_fa_one
       - resolved_turbulent_ke_ra_one
       - resolved_turbulent_ke_fa_one
       - reynolds_stress

    restart:
      restart_data_base_name: heliumPlume.rst
      restart_frequency: 2 
      restart_start: 2
      compression_level: 9
      compression_shuffle: yes 
    
  - name: ioRealm
    mesh: ../../mesh/io_mesh.g
    type: input_output

    field_registration:
      specifications:
        - field_name: mixture_fraction
          target_name: block_1
          field_size: 1
          field_type: node_rank

        - field_name: reynolds_stress
          target_name: block_1
          field_size: 6
          field_type: node_rank

    output:
      output_data_base_name: io_results.e
      output_frequency: 2
      output_node_set: no
      output_variables:
       - mixture_fraction
       - reynolds_stress

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_time: 0.20
      time_step: 1.0e-3
      time_stepping_type: adaptive
      time_step_count: 0
      second_order_accuracy: no

      realms:
        - fluidRealm
        - ioRealm

      transfers:
        - xfer_fluid_io
