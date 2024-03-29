// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <ABLProfileFunction.h>

// basic c++
#include <cmath>

namespace sierra {
namespace nalu {

// ABLProfileFunction - base class
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ABLProfileFunction::ABLProfileFunction()
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ABLProfileFunction::~ABLProfileFunction()
{
  // nothing to do
}

//==========================================================================
// Class Definition
//==========================================================================
// StableABLProfileFunction - stably stratified ABL
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
StableABLProfileFunction::StableABLProfileFunction(
  const double gamma_m, const double gamma_h)
  : gamma_m_(gamma_m), gamma_h_(gamma_h)
{
  // nothing else to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
StableABLProfileFunction::~StableABLProfileFunction()
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- velocity ---------------------------------------------------------
//--------------------------------------------------------------------------
double
StableABLProfileFunction::velocity(const double znorm) const
{
  return -gamma_m_ * znorm;
}

//--------------------------------------------------------------------------
//-------- temperature ---------------------------------------------------------
//--------------------------------------------------------------------------
double
StableABLProfileFunction::temperature(const double znorm) const
{
  return -gamma_h_ * znorm;
}

//==========================================================================
// Class Definition
//==========================================================================
// UnstableABLProfileFunction - unstably stratified ABL
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
UnstableABLProfileFunction::UnstableABLProfileFunction(
  const double beta_m, const double beta_h)
  : beta_m_(beta_m), beta_h_(beta_h)
{
  pi_ = std::acos(-1.0);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
UnstableABLProfileFunction::~UnstableABLProfileFunction()
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- velocity ---------------------------------------------------------
//--------------------------------------------------------------------------
double
UnstableABLProfileFunction::velocity(const double znorm) const
{
  const double xarg = (1.0 - beta_m_ * znorm);
  const double xfun = std::pow(xarg, 0.25);
  double psi_m = 2.0 * std::log(0.5 * (1.0 + xfun)) +
                 std::log(0.5 * (1.0 + xfun * xfun)) - 2.0 * std::atan(xfun) +
                 0.5 * pi_;
  return psi_m;
}

//--------------------------------------------------------------------------
//-------- temperature ---------------------------------------------------------
//--------------------------------------------------------------------------
double
UnstableABLProfileFunction::temperature(const double znorm) const
{
  const double yarg = std::sqrt(1.0 - beta_h_ * znorm);
  double psi_h = std::log(0.5 * (1.0 + yarg));
  return psi_h;
}

//==========================================================================
// Class Definition
//==========================================================================
// NeutralABLProfileFunction - neutrally stratified ABL
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
NeutralABLProfileFunction::NeutralABLProfileFunction()
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
NeutralABLProfileFunction::~NeutralABLProfileFunction()
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- velocity ---------------------------------------------------------
//--------------------------------------------------------------------------
double
NeutralABLProfileFunction::velocity(const double /* znorm */) const
{
  return 0.0;
}

//--------------------------------------------------------------------------
//-------- temperature ---------------------------------------------------------
//--------------------------------------------------------------------------
double
NeutralABLProfileFunction::temperature(const double /* znorm */) const
{
  return 0.0;
}

} // namespace nalu
} // namespace sierra
