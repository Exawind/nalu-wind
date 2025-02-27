Property Evaluations
--------------------

Property specification is provided in the material model section of the
input file. Unity Lewis number assumptions for diffusive flux
coefficients for mass fraction and enthalpy are assumed.

Density
+++++++

At present, property evaluation for density is given by constant, single
mixture fraction-based, HDF5 tables, or ideal gas. For ideal gas, we
support a non-isothermal, non-uniform and even an acoustically
compressible form.

Viscosity
+++++++++

Property evaluation for viscosity is given by constant, single mixture
fraction-based, simple tables or Sutherland’s three coefficient as a
function of temperature. When mixtures are used, either by reference or
species transport, only a mass fraction-weighed approach is used.

Specific Heat
+++++++++++++

Property evaluation for specific heat is either constant of two-band
standard NASA polynomials; again species composition weighting are used
(either transported or reference).

Lame Properties
+++++++++++++++

Lame constants are either of type constant or for use in mesh
motion/smoothing geometric whereby the values are inversely proportional
to the dual volume.
