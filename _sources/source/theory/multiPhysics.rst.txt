Multi-Physics
-------------

The equation set required to support the energy sector is already
represented as a multiphysics application. However, in some common cases
of coupling including conjugate heat transfer and coupling to
participating media radiation, an operator split method may be
preferred. The general concept is to define multiple Nalu-Wind Realms that
each own the mesh on which the particular physics is solved. Surface-
and volume-based couplings are supported through linear interpolation of
the coupling parameters.

A typical CHT application involves the coupling of a thermal response
and fluid transport. The coupling occurs between the surface that shares
the thermal equation and static enthalpy equation. Moreover, coupling to
a PMR solve is a volume-based coupling. Multiple Realms are supported
with multiple transfers.

In Nalu-Wind, the method to achieve coupling in CHT or RTE coupled systems is
through the usage of the STK Transfer module. This allows for linear
interpolation between disparate meshes. Advanced conservative transfers
are being evaluated, however, are not yet implemented in the code base.
In general, the STK Transfer interface allows for this design point.

For FSI, the usage of the transfer module is also expected.
