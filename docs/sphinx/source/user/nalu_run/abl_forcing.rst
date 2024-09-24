ABL Forcing
````````````````````

.. inpfile:: abl_forcing

  ``abl_forcing`` allows the user to specify desired velocities
  and temperatures at different heights.
  These velocities and temperatures are enforced through the
  use of source in the momentum and enthalpy equations.
  The ``abl_forcing`` option needs to be specified in the
  ``momentum`` and/or ``enthalpy`` source blocks:

  .. code-block:: yaml

     - source_terms:
        momentum: abl_forcing
        enthalpy: abl_forcing

  This option allows the code to implement source terms
  in the momentum and/or enthalpy equations.
  A sample
  section is shown below

  .. code-block:: yaml

   abl_forcing:
     search_method: stk_kdtree
     search_tolerance: 0.0001
     search_expansion_factor: 1.5
     output_frequency: 1

     from_target_part: [fluid_part]

     momentum:
       type: computed
       relaxation_factor: 1.0
       heights: [250.0, 500.0, 750.0]
       target_part_format: "zplane_%06.1f"

       # The velocities at each plane
       # Each list include a time and the velocities for each plane
       # Notice that the total number of elements in each list will be
       # number of planes + 1
       velocity_x:
         - [0.0,      10.0, 5.0, 15.0]
         - [100000.0, 10.0, 5.0, 15.0]

       velocity_y:
         - [0.0,      0.0, 0.0, 0.0]
         - [100000.0, 0.0, 0.0, 0.0]

       velocity_z:
         - [0.0, 0.0, 0.0, 0.0]
         - [100000.0, 0.0, 0.0, 0.0]

     temperature:
       type: computed
       relaxation_factor: 1.0
       heights: [250.0, 500.0, 750.0]
       target_part_format: "zplane_%06.1f"

       temperature:
         - [0.0,      300.0, 280.0, 310.0]
         - [100000.0, 300.0, 280.0, 310.0]
.. note::

  The variables in the :inpfile:`abl_forcing` subsection are
  prefixed with ``abl_forcing.name`` but only the variable
  name after the period should appear in the input file.

.. inpfile:: abl_forcing.search_method

    This specifies the search method algorithm within the
    stk framework. The default option `stk_kdtree` is recommended.

.. inpfile:: abl_forcing.search_tolerance

    This is the tolerance specified for the
    `search_method` algorithm. A default value of 0.0001 is recommended.

.. inpfile:: abl_forcing.search_expansion_factor

    This option is related to the stk search algorithm.
    A value of 1.5 is recommended.

.. inpfile:: abl_forcing.output_frequency

    This is the frequency at which the source term is written
    to the output value. A value of 1 means the source term
    will be written to the output file every time-step.

.. note::

   There are now two options in the following inputs.
   The can be ``momentum`` and/or ``temperature``.

.. inpfile:: abl_forcing.momentum.computed

 This option allows the user to choose if a momentum source is computed
 from a desired velocity (``computed``) or if a user defined
 source term is directly
 applied into the momentum equation (``user_defined``).

.. inpfile:: abl_forcing.momentum.relaxation_factor

  This is a relaxation factor which can be used to under/over-relax
  the momentum source term.
  The default value is 1.

.. inpfile:: abl_forcing.momentum.heights

  This is a list containing the planes at which the forcing should
  be implemented. Each input is the height for that plane.
  This is the naming convention in the mesh file.

.. inpfile:: abl_forcing.momentum.target_part_format

  This is the format in which the planes are saved in the
  mesh file.

.. inpfile:: abl_forcing.momentum.velocity_x

  A set of lists containing the time in the first element,
  followed by the desired velocity at each plane in the
  :math:`x` direction.

.. inpfile:: abl_forcing.momentum.velocity_y

  A set of lists containing the time in the first element,
  followed by the desired velocity at each plane in the
  :math:`y` direction.


.. inpfile:: abl_forcing.momentum.velocity_z

  A set of lists containing the time in the first element,
  followed by the desired velocity at each plane in the
  :math:`z` direction.

.. note::

  The temperature has the same inputs as the momentum source
  (``abl_forcing.temperature.type``,
  ``abl_forcing.temperature.relaxation_factor``,
  ``abl_forcing.temperature.heights``, and
  ``abl_forcing.temperature.target_part_format``)
  which take the same options.

.. inpfile:: abl_forcing.temperature.temperature

  A set of lists containing the time in the first element,
  followed by the desired temperature at each plane.
