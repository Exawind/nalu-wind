// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>

#include <AlgTraits.h>

#include <NaluEnv.h>

#include <stk_util/util/ReportHandler.hpp>
#include <stk_topology/topology.hpp>

#include <iostream>

#include <cmath>
#include <limits>
#include <array>
#include <map>
#include <memory>

namespace sierra {
namespace nalu {

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
MasterElement::MasterElement(const double scaleToStandardIsoFac)
  : nDim_(0),
    nodesPerElement_(0),
    numIntPoints_(0),
    scaleToStandardIsoFac_(scaleToStandardIsoFac)
{
  // nothing else
}

//--------------------------------------------------------------------------
//-------- isoparametric_mapping -------------------------------------------
//--------------------------------------------------------------------------
double
MasterElement::isoparametric_mapping(
  const double b, const double a, const double xi) const
{
  return xi * (b - a) / 2.0 + (a + b) / 2.0;
}

//--------------------------------------------------------------------------
//-------- within_tolerance ------------------------------------------------
//--------------------------------------------------------------------------
bool
MasterElement::within_tolerance(const double& val, const double& tol) const
{
  return (std::abs(val) < tol);
}

//--------------------------------------------------------------------------
//-------- vector_norm_sq --------------------------------------------------
//--------------------------------------------------------------------------
double
MasterElement::vector_norm_sq(const double* vect, int len) const
{
  double norm_sq = 0.0;
  for (int i = 0; i < len; i++) {
    norm_sq += vect[i] * vect[i];
  }
  return norm_sq;
}

} // namespace nalu
} // namespace sierra
