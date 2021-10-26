// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MasterElementWork_h
#define MasterElementWork_h

#include <AlgTraits.h>
#include <NaluEnv.h>

namespace sierra{
namespace nalu{

KOKKOS_FUNCTION
void hex_scs_det(const int nelem, const double *cordel, double *area_vec);


} // namespace nalu
} // namespace Sierra

#endif
