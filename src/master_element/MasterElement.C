/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>

#include <master_element/MasterElementHO.h>
#include <master_element/MasterElementUtils.h>

#include <element_promotion/LagrangeBasis.h>
#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/QuadratureRule.h>
#include <AlgTraits.h>

#include <NaluEnv.h>
#include <FORTRAN_Proto.h>

#include <stk_util/util/ReportHandler.hpp>
#include <stk_topology/topology.hpp>

#include <iostream>

#include <cmath>
#include <limits>
#include <array>
#include <map>
#include <memory>

namespace sierra{
namespace nalu{

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
MasterElement::MasterElement()
  : nDim_(0),
    nodesPerElement_(0),
    numIntPoints_(0),
    scaleToStandardIsoFac_(1.0)
{
  // nothing else
}

//--------------------------------------------------------------------------
//-------- isoparametric_mapping -------------------------------------------
//--------------------------------------------------------------------------
double
MasterElement::isoparametric_mapping( 
  const double b,
  const double a,
  const double xi) const
{
  return xi*(b-a)/2.0 +(a+b)/2.0;
}

//--------------------------------------------------------------------------
//-------- within_tolerance ------------------------------------------------
//--------------------------------------------------------------------------
bool 
MasterElement::within_tolerance( const double & val, const double & tol ) const
{
  return (std::abs(val)<tol);
}

//--------------------------------------------------------------------------
//-------- vector_norm_sq --------------------------------------------------
//--------------------------------------------------------------------------
double 
MasterElement::vector_norm_sq( const double * vect, int len ) const
{
  double norm_sq = 0.0;
  for (int i=0; i<len; i++) {
    norm_sq += vect[i]*vect[i];
  }
  return norm_sq;
}

} // namespace nalu
} // namespace sierra
