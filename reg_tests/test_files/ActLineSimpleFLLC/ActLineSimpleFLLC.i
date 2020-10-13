# -*- mode: yaml -*-
# NALU-WIND PARAMETERS
# ------
Simulations:
- name: sim1
  time_integrator: ti_1
  optimizer: opt1

linear_solvers:

- name: solve_scalar
  type: tpetra
  method: gmres
  preconditioner: mt_sgs
  tolerance: 1e-12
  max_iterations: 200
  kspace: 50
  output_level: 0

- name: solve_cont
  type: tpetra
  method: gmres
  preconditioner: muelu
  tolerance: 1e-12
  max_iterations: 200
  kspace: 50
  output_level: 0
  muelu_xml_file_name: ../../xml/milestone.xml
  summarize_muelu_timer: no

realms:

- name: realm_1
  mesh: "generated:30x30x30|bbox:-15.265,-15.265,-15.265,15.265,15.265,15.265|sideset:xXyYzZ|show"
  use_edges: yes
  automatic_decomposition_type: rcb

  equation_systems:
    name: theEqSys
    max_iterations: 2

    solver_system_specification:
      velocity: solve_scalar
      pressure: solve_cont
      turbulent_ke: solve_scalar

    systems:

    - LowMachEOM:
        name: myLowMach
        max_iterations: 1
        convergence_tolerance: 1e-5
    - TurbKineticEnergy:
        name: myTke
        max_iterations: 1
        convergence_tolerance: 1.0e-5

  initial_conditions:

  - constant: ic_1
    target_name: block_1
    value:
      pressure: 0.0
      velocity: [1.0, 0.0, 0.0]

  material_properties:
    target_name: block_1
    specifications:
    - name: density
      type: constant
      value: 1.0

    - name: viscosity
      type: constant
      value: 1e-5

  boundary_conditions:

  - inflow_boundary_condition: bc_1
    target_name: surface_1
    inflow_user_data:
      velocity: [1.0, 0.0, 0.0]

  - open_boundary_condition: bc_2
    target_name: surface_2
    open_user_data:
      pressure: 0.0
      velocity: [1.0, 0.0, 0.0]

  - symmetry_boundary_condition: bc_3
    target_name: surface_3
    symmetry_user_data:

  - symmetry_boundary_condition: bc_4
    target_name: surface_4
    symmetry_user_data:

  - symmetry_boundary_condition: bc_5
    target_name: surface_5
    symmetry_user_data:

  - symmetry_boundary_condition: bc_6
    target_name: surface_6
    symmetry_user_data:

  solution_options:
    name: myOptions
    turbulence_model: ksgs

    options:
    # Model constants for the 1-eq k SGS model.
    - turbulence_model_constants:
        kappa: 0.4
        cEps: 0.93
        cmuEps: 0.0673

    - laminar_prandtl:
        enthalpy: 0.7

    # Turbulent Prandtl number is 1/3 following Moeng (1984).
    - turbulent_prandtl:
        enthalpy: 0.3333

    # SGS viscosity is divided by Schmidt number in the k SGS diffusion
    # term.  In Moeng (1984), SGS viscosity is multiplied by 2, hence
    # we divide by 1/2
    - turbulent_schmidt:
        turbulent_ke: 0.5

    - hybrid_factor:
        velocity: 1.0

    - limiter:
        pressure: no
        velocity: no

    - projected_nodal_gradient:
        pressure: element
        velocity: element

    - source_terms:
        momentum: 
          - actuator
        turblent_ke:
          - rodi

  actuator:
    type: ActLineSimpleNGP
    search_method: stk_kdtree
    search_target_part: block_1
    fllt_correction: yes

    n_simpleblades: 1
    debug_output: no
    Blade0:
      num_force_pts_blade: 50
      output_file_name: blade_dump.csv
      epsilon: [3.0, 3.0, 3.0]
      p1: [0, -6.25, 0] 
      p2: [0,  6.25, 0]
      p1_zero_alpha_dir: [1, 0, 0]
      chord_table: [1.0]
      twist_table: [6.0]
      aoa_table: [0.0, 20.0]
      cl_table:  [0.0, 2.19]
      cd_table:  [0]

  output:
    output_data_base_name: output/alm_uniform_inflow.exo
    output_frequency: 1
    output_node_set: no
    output_variables:
    - velocity
    - pressure
    - actuator_source

  data_probes:  
    output_format: text 
    output_frequency: 1
    search_method: stk_kdtree
    search_tolerance: 1.0e-5 #1.0e-3
    search_expansion_factor: 2.0

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
    termination_step_count: 100
    time_step: 0.0125
    time_stepping_type: fixed
    time_step_count: 0
    second_order_accuracy: no

    realms:
    - realm_1
