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
    muelu_xml_file_name: ../../xml/milestone_aspect_ratio.xml

realms:

  - name: realm_1
    mesh: ../../mesh/waterChannel_mks.g
    use_edges: no

    equation_systems:
      name: theEqSys
      max_iterations: 4

      solver_system_specification:
        velocity: solve_scalar
        pressure: solve_cont
        enthalpy: solve_scalar

      systems:

        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-5

        - Enthalpy:
            name: myEnth
            max_iterations: 1
            convergence_tolerance: 1e-5

    material_properties:

      target_name: block_10

      specifications:
 
        - name: density
          type: generic
          generic_property_evaluator_name: water_density_T

        - name: viscosity
          type: generic
          generic_property_evaluator_name: water_viscosity_T

        - name: specific_heat
          type: generic
          generic_property_evaluator_name: water_specific_heat_T

        - name: thermal_conductivity
          type: generic
          generic_property_evaluator_name: water_thermal_conductivity_T

    initial_conditions:
      - constant: ic_1
        target_name: block_10
        value:
          pressure: 0
          velocity: [0,0]  
          temperature: 283.16
  
    boundary_conditions:

    - inflow_boundary_condition: bc_inflow
      target_name: surface_1
      inflow_user_data:
        velocity: [1.148, 0.0]
        temperature: 283.16

    - open_boundary_condition: bc_open
      target_name: surface_2
      open_user_data:
        velocity: [0,0]
        pressure: 0.0
        temperature: 283.16

    - wall_boundary_condition: bc_lower
      target_name: surface_3
      wall_user_data:
        velocity: [0,0]
        temperature: 318.0

    - wall_boundary_condition: bc_upper
      target_name: surface_4
      wall_user_data:
        velocity: [0,0]
        use_wall_function: yes

    - wall_boundary_condition: bc_cylinder
      target_name: surface_5
      wall_user_data:
        velocity: [0,0]
        user_function_name:
         temperature: flow_past_cylinder

    solution_options:
      name: myOptions
      turbulence_model: wale
      interp_rhou_together_for_mdot: yes

      use_consolidated_solver_algorithm: yes
      use_consolidated_face_elem_bc_algorithm: yes

      options:

        - turbulent_prandtl:
            enthalpy: 0.90

        - user_constants:
            gravity: [0.0,-9.81]
            reference_density: 998.4103224

        - limiter:
            pressure: no
            velocity: no
            enthalpy: yes 

        - peclet_function_form:
            velocity: tanh
            enthalpy: tanh

        - peclet_function_tanh_transition:
            velocity: 5000.0
            enthalpy: 2.01

        - peclet_function_tanh_width:
            velocity: 200.0
            enthalpy: 4.02

        - element_source_terms:
            momentum: [lumped_momentum_time_derivative, upw_advection_diffusion, buoyancy]
            continuity: [lumped_density_time_derivative, advection]
            enthalpy: [lumped_enthalpy_time_derivative, upw_advection_diffusion]

    turbulence_averaging:
      time_filter_interval: 100000.0

      specifications:

        - name: one
          target_name: block_10
          reynolds_averaged_variables:
            - velocity
            - enthalpy
            - resolved_turbulent_ke

          favre_averaged_variables:
            - velocity
            - enthalpy
            - resolved_turbulent_ke

          compute_tke: yes 
          compute_reynolds_stress: yes

        - name: two
          target_name: surface_5
          reynolds_averaged_variables:
            - normal_heat_flux
          favre_averaged_variables:
            - normal_heat_flux

    output:
      output_data_base_name: heatedWaterChannelElem.e
      output_frequency: 5
      output_node_set: no
      output_variables:
       - velocity
       - velocity_ra_one
       - velocity_fa_one
       - pressure
       - enthalpy
       - enthalpy_ra_one
       - enthalpy_fa_one
       - temperature
       - specific_heat
       - thermal_conductivity
       - viscosity
       - normal_heat_flux
       - normal_heat_flux_ra_two
       - normal_heat_flux_fa_two
       - resolved_turbulent_ke_ra_one
       - resolved_turbulent_ke_fa_one
       - reynolds_stress

    restart:
      restart_data_base_name: heatedWaterChannelElem.rst
      output_frequency: 10

Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      termination_step_count: 20
      time_step: 1.0e-5
      time_stepping_type: adaptive
      time_step_count: 0
      second_order_accuracy: yes

      realms:
        - realm_1
