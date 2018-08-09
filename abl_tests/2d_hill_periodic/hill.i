Simulations:
  - name: sim1
    time_integrator: ti_1
    optimizer: opt1


# Specify the linear system solvers.
linear_solvers:

  # solver for scalar equations
  - name: solve_scalar
    type: tpetra
    method: gmres
    preconditioner: sgs
    tolerance: 1e-6
    max_iterations: 75
    kspace: 75
    output_level: 0

  # solver for the pressure Poisson equation
  - name: solve_cont
    type: tpetra
    method: gmres
    preconditioner: muelu
    tolerance: 1e-6
    max_iterations: 75
    kspace: 75
    output_level: 0
    recompute_preconditioner: no
    muelu_xml_file_name: ./milestone.xml


# Specify the differnt physics realms.  Here, we have one for the fluid and one for io transfer to south/west inflow planes.
realms:

  # The fluid realm that uses the 2 km x 0.5 km Gaussian hill mesh.
  - name: fluidRealm
    mesh: ./hill.exo
    use_edges: yes
    automatic_decomposition_type: rcb

    # This defines the equations to be solved: momentum, pressure, static enthalpy, 
    # and subgrid-scale turbulent kinetic energy.  The equation system will be iterated
    # a maximum of 4 outer iterations.
    equation_systems:
      name: theEqSys
      max_iterations: 4

      # This defines which solver to use for each equation set.  See the
      # "linear_solvers" block.  All use the scalar solver, except pressure.
      solver_system_specification:
        velocity: solve_scalar
        pressure: solve_cont

      # This defines the equation systems, maximum number of inner iterations,
      # and scaled nonlinear residual tolerance.
      systems:

        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1.0e-5

    # Specify the properties of the fluid, in this case air.
    material_properties:

#      target_name: [Unspecified-2-Hex]
      target_name: [fluid]

      constant_specification:
       universal_gas_constant: 8314.4621
       reference_pressure: 101325.0

      reference_quantities:
        - species_name: Air
          mw: 29.0
          mass_fraction: 1.0

      specifications:
 
        # Density here was computed such that P_ref = rho_ref*(R/mw)*300K
        - name: density
          type: constant
          value: 1.0

        - name: viscosity
          type: constant
          value: 1.0E-5

        - name: specific_heat
          type: constant
          value: 1000.0

    # The initial conditions are that pressure is uniformly 0 Pa and velocity
    # is 1 m/s from the west.
    initial_conditions:
      - constant: ic_1
#        target_name: [Unspecified-2-Hex]
        target_name: [fluid]
        value:
          pressure: 0.0
          velocity: [1.0, 0.0, 0.0]


    # Boundary conditions are periodic on the north, south, east, and west
    # sides.  The lower boundary condition is a slip wall.  The upper boundary 
    # is an inflow/outflow surface where the velocity field is determined
    # via a potential flow model for a slab near the upper boundary.
    boundary_conditions:

    - periodic_boundary_condition: bc_north_south
      target_name: [north, south]
      periodic_user_data:
        search_tolerance: 0.0001

    - periodic_boundary_condition: bc_east_west
      target_name: [east, west]
      periodic_user_data:
        search_tolerance: 0.0001

    - abltop_boundary_condition: bc_upper
      target_name: top
      abltop_user_data:
        potential_flow_bc: true
        grid_dimensions: [121, 2, 61]

    - symmetry_boundary_condition: bc_lower
      target_name: terrain
      symmetry_user_data:

    solution_options:
      name: myOptions
      interp_rhou_together_for_mdot: yes

      options:
        - limiter:
            pressure: no
            velocity: no
            enthalpy: yes 

        - peclet_function_form:
            velocity: classic
            enthalpy: tanh
            turbulent_ke: tanh

        - peclet_function_tanh_transition:
            velocity: 50000.0
            enthalpy: 2.0
            turbulent_ke: 2.0

        - peclet_function_tanh_width:
            velocity: 200.0
            enthalpy: 1.0
            turbulent_ke: 1.0

    output:
      output_data_base_name: hill.e
      output_frequency: 1
      output_nodse_set: no
      output_variables:
       - velocity
       - pressure

    restart:
      restart_data_base_name: hill.rst
      restart_node_set: true
#      restart_time: 0.4
      restart_frequency: 20
      restart_start: 20

# This defines the time step size, count, etc.
Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0.0
      termination_step_count: 10
      time_step: 0.02
      time_stepping_type: fixed
      time_step_count: 0
      second_order_accuracy: yes

      realms:
        - fluidRealm
