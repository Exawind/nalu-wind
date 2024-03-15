// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <property_evaluator/PropertyEvaluator.h>
#include <property_evaluator/SutherlandsPropertyEvaluator.h>
#include <property_evaluator/ReferencePropertyData.h>

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

#include <cmath>
#include <stdexcept>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// SutherlandsPropertyEvaluator - evaluates mu based on temperature; Ykref
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
SutherlandsPropertyEvaluator::SutherlandsPropertyEvaluator(
  const std::map<std::string, ReferencePropertyData*>& referencePropertyDataMap,
  const std::map<std::string, std::vector<double>>& polynomialCoeffsMap)
  : PropertyEvaluator()
{
  // extract the reference property data size
  size_t refPropSize = referencePropertyDataMap.size();

  // resize reference mass fraction and polynomial size
  refMassFraction_.resize(refPropSize);
  polynomialCoeffs_.resize(refPropSize);

  // increment on iterator
  size_t k = 0;

  // save off reference values for yk
  std::map<std::string, ReferencePropertyData*>::const_iterator itrp;
  for (itrp = referencePropertyDataMap.begin();
       itrp != referencePropertyDataMap.end(); ++itrp, ++k) {
    ReferencePropertyData* propData = (*itrp).second;
    refMassFraction_[k] = propData->massFraction_;
  }

  // save off polynomial coeffs
  k = 0;
  std::map<std::string, std::vector<double>>::const_iterator itpc;
  for (itpc = polynomialCoeffsMap.begin(); itpc != polynomialCoeffsMap.end();
       ++itpc, ++k) {
    std::vector<double> polyVec = (*itpc).second;
    size_t polyVecSize = polyVec.size();
    if (polyVecSize < 3)
      throw std::runtime_error(
        "Sutherlands polynomial evaluator needs three coeffs:");
    polynomialCoeffs_[k].resize(polyVecSize);
    double* pt_poly = &polynomialCoeffs_[k][0];
    for (size_t j = 0; j < polyVecSize; ++j) {
      pt_poly[j] = polyVec[j];
    }
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
SutherlandsPropertyEvaluator::~SutherlandsPropertyEvaluator()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
double
SutherlandsPropertyEvaluator::execute(
  double* indVarList, stk::mesh::Entity /*node*/)
{
  const double T = indVarList[0];

  // extract size and process sum
  const size_t ykSize = refMassFraction_.size();
  double sum_mu = 0.0;
  for (size_t k = 0; k < ykSize; ++k) {
    sum_mu +=
      refMassFraction_[k] * compute_viscosity(T, &polynomialCoeffs_[k][0]);
  }

  return sum_mu;
}

//--------------------------------------------------------------------------
//-------- compute_viscosity -----------------------------------------------
//--------------------------------------------------------------------------
double
SutherlandsPropertyEvaluator::compute_viscosity(
  const double& T, const double* pt_poly)
{
  const double muRef = pt_poly[0];
  const double TRef = pt_poly[1];
  const double SRef = pt_poly[2];

  return muRef * std::pow(T / TRef, 1.5) * (TRef + SRef) / (T + SRef);
}

//==========================================================================
// Class Definition
//==========================================================================
// SutherlandsYkPropertyEvaluator - evaluates mu based on temperature; Yk
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
SutherlandsYkPropertyEvaluator::SutherlandsYkPropertyEvaluator(
  const std::map<std::string, std::vector<double>>& polynomialCoeffsMap,
  stk::mesh::MetaData& metaData)
  : PropertyEvaluator(), massFraction_(NULL), ykVecSize_(0)
{
  // sizing
  ykVecSize_ = polynomialCoeffsMap.size();
  polynomialCoeffs_.resize(ykVecSize_);

  // save off mass fraction field
  massFraction_ =
    metaData.get_field<double>(stk::topology::NODE_RANK, "mass_fraction");

  // save off polynomial coeffs
  size_t k = 0;
  std::map<std::string, std::vector<double>>::const_iterator itpc;
  for (itpc = polynomialCoeffsMap.begin(); itpc != polynomialCoeffsMap.end();
       ++itpc, ++k) {
    std::vector<double> polyVec = (*itpc).second;
    size_t polyVecSize = polyVec.size();
    if (polyVecSize != 3)
      throw std::runtime_error(
        "viscosity polynomial evaluator needs 3 coeffs:");
    polynomialCoeffs_[k].resize(polyVecSize);
    double* pt_poly = &polynomialCoeffs_[k][0];
    for (size_t j = 0; j < polyVecSize; ++j) {
      pt_poly[j] = polyVec[j];
    }
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
SutherlandsYkPropertyEvaluator::~SutherlandsYkPropertyEvaluator()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
double
SutherlandsYkPropertyEvaluator::execute(
  double* indVarList, stk::mesh::Entity node)
{
  const double T = indVarList[0];
  const double* massFraction = stk::mesh::field_data(*massFraction_, node);

  // process sum
  double sum_mu = 0.0;
  for (size_t k = 0; k < ykVecSize_; ++k) {
    sum_mu += massFraction[k] * compute_viscosity(T, &polynomialCoeffs_[k][0]);
  }

  return sum_mu;
}

//--------------------------------------------------------------------------
//-------- compute_viscosity -----------------------------------------------
//--------------------------------------------------------------------------
double
SutherlandsYkPropertyEvaluator::compute_viscosity(
  const double& T, const double* pt_poly)
{
  const double muRef = pt_poly[0];
  const double TRef = pt_poly[1];
  const double SRef = pt_poly[2];

  return muRef * std::pow(T / TRef, 1.5) * (TRef + SRef) / (T + SRef);
}

//==========================================================================
// Class Definition
//==========================================================================
// SutherlandsYkTrefPropertyEvaluator - evaluates mu based on Yk and Tref
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
SutherlandsYkTrefPropertyEvaluator::SutherlandsYkTrefPropertyEvaluator(
  const std::map<std::string, std::vector<double>>& polynomialCoeffsMap,
  stk::mesh::MetaData& metaData,
  const double tRef)
  : SutherlandsYkPropertyEvaluator(polynomialCoeffsMap, metaData), tRef_(tRef)
{
  // base class handles everything that is required
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
SutherlandsYkTrefPropertyEvaluator::~SutherlandsYkTrefPropertyEvaluator()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
double
SutherlandsYkTrefPropertyEvaluator::execute(
  double* /*indVarList*/, stk::mesh::Entity node)
{
  const double T = tRef_;
  const double* massFraction = stk::mesh::field_data(*massFraction_, node);

  // process sum
  double sum_mu = 0.0;
  for (size_t k = 0; k < ykVecSize_; ++k) {
    sum_mu += massFraction[k] * compute_viscosity(T, &polynomialCoeffs_[k][0]);
  }

  return sum_mu;
}

} // namespace nalu
} // namespace sierra
