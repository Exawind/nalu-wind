// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <property_evaluator/PropertyEvaluator.h>
#include <property_evaluator/IdealGasPropertyEvaluator.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>

#include <vector>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// IdealGasTPropertyEvaluator - evaluates density as a function of T
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
IdealGasTPropertyEvaluator::IdealGasTPropertyEvaluator(
  double pRef,
  double universalR,
  std::vector<std::pair<double, double>> mwMassFracVec)
  : PropertyEvaluator(), pRef_(pRef), R_(universalR), mw_(0.0)
{
  // compute mixture mw
  double sum = 0.0;
  for (std::size_t k = 0; k < mwMassFracVec.size(); ++k) {
    sum += mwMassFracVec[k].second / mwMassFracVec[k].first;
  }
  mw_ = 1.0 / sum;
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
IdealGasTPropertyEvaluator::~IdealGasTPropertyEvaluator()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
double
IdealGasTPropertyEvaluator::execute(
  double* indVarList, stk::mesh::Entity /*node*/)
{
  const double T = indVarList[0];
  return pRef_ * mw_ / R_ / T;
}

//==========================================================================
// Class Definition
//==========================================================================
// IdealGasTYkPropertyEvaluator - evaluates density as a function of T and Yk
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
IdealGasTYkPropertyEvaluator::IdealGasTYkPropertyEvaluator(
  double pRef,
  double universalR,
  std::vector<double> mwVec,
  stk::mesh::MetaData& metaData)
  : PropertyEvaluator(),
    pRef_(pRef),
    R_(universalR),
    massFraction_(NULL),
    mwVecSize_(0)
{
  // sizing
  mwVecSize_ = mwVec.size();
  mwVec_.resize(mwVecSize_);

  // save off mwVec (reference quantity)
  for (std::size_t k = 0; k < mwVecSize_; ++k) {
    mwVec_[k] = mwVec[k];
  }

  // save off mass fraction field
  massFraction_ =
    metaData.get_field<double>(stk::topology::NODE_RANK, "mass_fraction");
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
IdealGasTYkPropertyEvaluator::~IdealGasTYkPropertyEvaluator()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
double
IdealGasTYkPropertyEvaluator::execute(
  double* indVarList, stk::mesh::Entity node)
{
  const double T = indVarList[0];
  const double* massFraction = stk::mesh::field_data(*massFraction_, node);
  const double mw = compute_mw(massFraction);
  return pRef_ * mw / R_ / T;
}

//--------------------------------------------------------------------------
//-------- compute_mw ------------------------------------------------------
//--------------------------------------------------------------------------
double
IdealGasTYkPropertyEvaluator::compute_mw(const double* massFraction)
{

  // compute mixture mw
  double sum = 0.0;
  for (std::size_t k = 0; k < mwVecSize_; ++k) {
    sum += massFraction[k] / mwVec_[k];
  }
  return 1.0 / sum;
}

//==========================================================================
// Class Definition
//==========================================================================
// IdealGasTPPropertyEvaluator - evaluates density as a function of T and P
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
IdealGasTPPropertyEvaluator::IdealGasTPPropertyEvaluator(
  double universalR,
  std::vector<std::pair<double, double>> mwMassFracVec,
  stk::mesh::MetaData& metaData)
  : PropertyEvaluator(), R_(universalR), mw_(0.0), pressure_(NULL)
{

  // save off mass fraction field
  pressure_ = metaData.get_field<double>(stk::topology::NODE_RANK, "pressure");

  // compute mixture mw
  double sum = 0.0;
  for (std::size_t k = 0; k < mwMassFracVec.size(); ++k) {
    sum += mwMassFracVec[k].second / mwMassFracVec[k].first;
  }
  mw_ = 1.0 / sum;
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
IdealGasTPPropertyEvaluator::~IdealGasTPPropertyEvaluator()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
double
IdealGasTPPropertyEvaluator::execute(double* indVarList, stk::mesh::Entity node)
{
  const double T = indVarList[0];
  const double P = *stk::mesh::field_data(*pressure_, node);
  return P * mw_ / R_ / T;
}

//==========================================================================
// Class Definition
//==========================================================================
// IdealGasYkPropertyEvaluator - evaluates density as a function of Yk
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
IdealGasYkPropertyEvaluator::IdealGasYkPropertyEvaluator(
  double pRef,
  double tRef,
  double universalR,
  std::vector<double> mwVec,
  stk::mesh::MetaData& metaData)
  : PropertyEvaluator(),
    pRef_(pRef),
    tRef_(tRef),
    R_(universalR),
    massFraction_(NULL),
    mwVecSize_(0)
{
  // sizing
  mwVecSize_ = mwVec.size();
  mwVec_.resize(mwVecSize_);

  // save off mwVec (reference quantity)
  for (std::size_t k = 0; k < mwVecSize_; ++k) {
    mwVec_[k] = mwVec[k];
  }

  // save off mass fraction field
  massFraction_ =
    metaData.get_field<double>(stk::topology::NODE_RANK, "mass_fraction");
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
IdealGasYkPropertyEvaluator::~IdealGasYkPropertyEvaluator()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
double
IdealGasYkPropertyEvaluator::execute(
  double* /*indVarList*/, stk::mesh::Entity node)
{
  const double* massFraction = stk::mesh::field_data(*massFraction_, node);
  const double mw = compute_mw(massFraction);
  return pRef_ * mw / R_ / tRef_;
}

//--------------------------------------------------------------------------
//-------- compute_mw ------------------------------------------------------
//--------------------------------------------------------------------------
double
IdealGasYkPropertyEvaluator::compute_mw(const double* massFraction)
{

  // compute mixture mw
  double sum = 0.0;
  for (std::size_t k = 0; k < mwVecSize_; ++k) {
    sum += massFraction[k] / mwVec_[k];
  }
  return 1.0 / sum;
}

} // namespace nalu
} // namespace sierra
