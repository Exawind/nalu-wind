// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <property_evaluator/PolynomialPropertyEvaluator.h>
#include <property_evaluator/PropertyEvaluator.h>
#include <property_evaluator/ReferencePropertyData.h>

#include <stdexcept>

namespace sierra{
namespace nalu{


//==========================================================================
// Class Definition
//==========================================================================
// PolynomialPropertyEvaluator - evaluates property based on polynomial fit
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
PolynomialPropertyEvaluator::PolynomialPropertyEvaluator(
    const std::map<std::string, ReferencePropertyData*> &referencePropertyDataMap,
    const std::map<std::string, std::vector<double> > &lowPolynomialCoeffsMap,
    const std::map<std::string, std::vector<double> > &highPolynomialCoeffsMap,
    double universalR)
  : PropertyEvaluator(),
    universalR_(universalR),
    ykVecSize_(referencePropertyDataMap.size()),
    TlowHigh_(1000.0)
{
  // sizing
  mw_.resize(ykVecSize_);
  lowPolynomialCoeffs_.resize(ykVecSize_);
  highPolynomialCoeffs_.resize(ykVecSize_);
  
  // increment on iterator
  size_t k = 0;

  // save off reference values for mw_
  std::map<std::string, ReferencePropertyData*>::const_iterator itrp;
  for ( itrp = referencePropertyDataMap.begin();
        itrp!= referencePropertyDataMap.end(); ++itrp, ++k) {
    ReferencePropertyData *propData = (*itrp).second;
    mw_[k] = propData->mw_;
  }

  // save off polynomial coeffs; high/low
  k = 0;
  std::map<std::string, std::vector<double> >::const_iterator itpc;
  for ( itpc = lowPolynomialCoeffsMap.begin();
        itpc!= lowPolynomialCoeffsMap.end(); ++itpc, ++k) {
    std::vector<double> polyVec= (*itpc).second;
    size_t polyVecSize = polyVec.size();
    lowPolynomialCoeffs_[k].resize(polyVecSize);
    double *pt_poly = &lowPolynomialCoeffs_[k][0];
    for ( size_t j = 0; j < polyVecSize; ++j ) {
      pt_poly[j] = polyVec[j];
    }
  }
  
  k = 0;
  for ( itpc = highPolynomialCoeffsMap.begin();
        itpc!= highPolynomialCoeffsMap.end(); ++itpc, ++k) {
    std::vector<double> polyVec= (*itpc).second;
    size_t polyVecSize = polyVec.size();
    highPolynomialCoeffs_[k].resize(polyVecSize);
    double *pt_poly = &highPolynomialCoeffs_[k][0];
    for ( size_t j = 0; j < polyVecSize; ++j ) {
      pt_poly[j] = polyVec[j];
   }
  }

}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
PolynomialPropertyEvaluator::~PolynomialPropertyEvaluator()
{
  // nothing
}

} // namespace nalu
} // namespace Sierra
