# -*- mode: yaml -*-
#
# Example Nalu input file for a heat conduction problem
#

Simulations:
  - name: sim1
    time_integrator: ti_1
    optimizer: opt1

linear_solvers:
  - name: solve_scalar
    type: tpetra
    method: gmres 
    preconditioner: sgs 
    tolerance: 1e-3
    max_iterations: 75 
    kspace: 75 
    output_level: 0

realms:

  - name: realm_1
    mesh: periodic3d.g
    use_edges: no 
    automatic_decomposition_type: rcb

    equation_systems:
      name: theEqSys
      max_iterations: 2 
  
      solver_system_specification:
        temperature: solve_scalar
   
      systems:
        - HeatConduction:
            name: myHC
            max_iterations: 1
            convergence_tolerance: 1e-5

    initial_conditions:

      - constant: ic_1
        target_name: block_1
        value:
         temperature: 10.0

    material_properties:
      target_name: block_1
      specifications:
        - name: density
          type: constant
          value: 1.0
        - name: thermal_conductivity
          type: constant
          value: 1.0
        - name: specific_heat
          type: constant
          value: 1.0

    boundary_conditions:

    - wall_boundary_condition: bc_left
      target_name: surface_1
      wall_user_data:
        temperature: 20.0


    - wall_boundary_condition: bc_right
      target_name: surface_2
      wall_user_data:
        temperature: 40.0

    solution_options:
      name: myOptions

      use_consolidated_solver_algorithm: yes

      options:
      - element_source_terms:
          temperature: FEM_DIFF

    output:
      output_data_base_name: femHC.e
      output_frequency: 10
      output_node_set: no 
      output_variables:
       - dual_nodal_volume
       - temperature

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 25 
      time_step: 10.0 
      time_stepping_type: fixed
      time_step_count: 0
      second_order_accuracy: no

      realms:
        - realm_1
