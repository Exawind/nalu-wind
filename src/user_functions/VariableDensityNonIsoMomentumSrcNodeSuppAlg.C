// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/VariableDensityNonIsoMomentumSrcNodeSuppAlg.h>
#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <SolutionOptions.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// VariableDensityNonIsoMomentumSrcNodeSuppAlg - base class for algorithm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
VariableDensityNonIsoMomentumSrcNodeSuppAlg::
  VariableDensityNonIsoMomentumSrcNodeSuppAlg(Realm& realm)
  : SupplementalAlgorithm(realm),
    coordinates_(NULL),
    dualNodalVolume_(NULL),
    nDim_(realm_.spatialDimension_),
    unot_(1.0),
    vnot_(1.0),
    wnot_(1.0),
    pnot_(1.0),
    hnot_(1.0),
    a_(20.0),
    ah_(10.0),
    visc_(0.00125),
    Pref_(100.0),
    MW_(30.0),
    R_(10.0),
    Tref_(300.0),
    Cp_(0.01),
    pi_(std::acos(-1.0)),
    twoThirds_(2.0 / 3.0 * realm_.get_divU()),
    rhoRef_(1.0),
    gx_(0.0),
    gy_(0.0),
    gz_(0.0)
{
  // save off fields
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<double>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name());
  dualNodalVolume_ =
    meta_data.get_field<double>(stk::topology::NODE_RANK, "dual_nodal_volume");

  // internal source
  srcXi_.resize(nDim_);

  // extract user parameters from solution options
  std::vector<double> gravity = realm_.solutionOptions_->gravity_;
  rhoRef_ = realm_.solutionOptions_->referenceDensity_;
  gx_ = gravity[0];
  gy_ = gravity[1];
  gz_ = gravity[2];
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
VariableDensityNonIsoMomentumSrcNodeSuppAlg::setup()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
VariableDensityNonIsoMomentumSrcNodeSuppAlg::node_execute(
  double* /*lhs*/, double* rhs, stk::mesh::Entity node)
{
  // deal with lumped mass matrix
  const double* coords = stk::mesh::field_data(*coordinates_, node);
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node);

  const double x = coords[0];
  const double y = coords[1];
  const double z = coords[2];

  srcXi_[0] =
    Pref_ * MW_ / R_ *
      pow(
        hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
            Cp_ +
          Tref_,
        -0.2e1) *
      unot_ * unot_ * pow(cos(a_ * pi_ * x), 0.2e1) *
      pow(sin(a_ * pi_ * y), 0.2e1) * pow(sin(a_ * pi_ * z), 0.2e1) * hnot_ *
      sin(ah_ * pi_ * x) * ah_ * pi_ * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
      Cp_ -
    0.2e1 * Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      unot_ * unot_ * cos(a_ * pi_ * x) * pow(sin(a_ * pi_ * y), 0.2e1) *
      pow(sin(a_ * pi_ * z), 0.2e1) * sin(a_ * pi_ * x) * a_ * pi_ -
    Pref_ * MW_ / R_ *
      pow(
        hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
            Cp_ +
          Tref_,
        -0.2e1) *
      vnot_ * sin(a_ * pi_ * x) * cos(a_ * pi_ * y) *
      pow(sin(a_ * pi_ * z), 0.2e1) * unot_ * cos(a_ * pi_ * x) *
      sin(a_ * pi_ * y) * hnot_ * cos(ah_ * pi_ * x) * sin(ah_ * pi_ * y) *
      ah_ * pi_ * cos(ah_ * pi_ * z) / Cp_ +
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      vnot_ * sin(a_ * pi_ * x) * pow(sin(a_ * pi_ * y), 0.2e1) * a_ * pi_ *
      pow(sin(a_ * pi_ * z), 0.2e1) * unot_ * cos(a_ * pi_ * x) -
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      vnot_ * sin(a_ * pi_ * x) * pow(cos(a_ * pi_ * y), 0.2e1) *
      pow(sin(a_ * pi_ * z), 0.2e1) * unot_ * cos(a_ * pi_ * x) * a_ * pi_ +
    Pref_ * MW_ / R_ *
      pow(
        hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
            Cp_ +
          Tref_,
        -0.2e1) *
      wnot_ * sin(a_ * pi_ * x) * pow(sin(a_ * pi_ * y), 0.2e1) *
      cos(a_ * pi_ * z) * unot_ * cos(a_ * pi_ * x) * sin(a_ * pi_ * z) *
      hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * sin(ah_ * pi_ * z) *
      ah_ * pi_ / Cp_ -
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      wnot_ * sin(a_ * pi_ * x) * pow(sin(a_ * pi_ * y), 0.2e1) *
      pow(sin(a_ * pi_ * z), 0.2e1) * a_ * pi_ * unot_ * cos(a_ * pi_ * x) +
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      wnot_ * sin(a_ * pi_ * x) * pow(sin(a_ * pi_ * y), 0.2e1) *
      pow(cos(a_ * pi_ * z), 0.2e1) * unot_ * cos(a_ * pi_ * x) * a_ * pi_ -
    visc_ * (-(unot_ * cos(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
                 sin(a_ * pi_ * y) * sin(a_ * pi_ * z) -
               vnot_ * cos(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
                 sin(a_ * pi_ * y) * sin(a_ * pi_ * z) +
               wnot_ * cos(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
                 sin(a_ * pi_ * y) * sin(a_ * pi_ * z)) *
               twoThirds_ +
             0.2e1 * unot_ * cos(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
               sin(a_ * pi_ * y) * sin(a_ * pi_ * z)) -
    visc_ * (unot_ * cos(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
               sin(a_ * pi_ * y) * sin(a_ * pi_ * z) -
             vnot_ * cos(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
               sin(a_ * pi_ * y) * sin(a_ * pi_ * z)) -
    visc_ * (unot_ * cos(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
               sin(a_ * pi_ * y) * sin(a_ * pi_ * z) +
             wnot_ * cos(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
               sin(a_ * pi_ * y) * sin(a_ * pi_ * z)) +
    0.50e0 * pnot_ * sin(0.2e1 * a_ * pi_ * x) * a_ * pi_ -
    (Pref_ * MW_ / R_ /
       (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
          Cp_ +
        Tref_) -
     rhoRef_) *
      gx_;

  srcXi_[1] =
    -Pref_ * MW_ / R_ *
      pow(
        hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
            Cp_ +
          Tref_,
        -0.2e1) *
      vnot_ * sin(a_ * pi_ * x) * cos(a_ * pi_ * y) *
      pow(sin(a_ * pi_ * z), 0.2e1) * unot_ * cos(a_ * pi_ * x) *
      sin(a_ * pi_ * y) * hnot_ * sin(ah_ * pi_ * x) * ah_ * pi_ *
      cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) / Cp_ -
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      vnot_ * pow(cos(a_ * pi_ * x), 0.2e1) * a_ * pi_ * cos(a_ * pi_ * y) *
      pow(sin(a_ * pi_ * z), 0.2e1) * unot_ * sin(a_ * pi_ * y) +
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      vnot_ * pow(sin(a_ * pi_ * x), 0.2e1) * cos(a_ * pi_ * y) *
      pow(sin(a_ * pi_ * z), 0.2e1) * unot_ * a_ * pi_ * sin(a_ * pi_ * y) +
    Pref_ * MW_ / R_ *
      pow(
        hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
            Cp_ +
          Tref_,
        -0.2e1) *
      vnot_ * vnot_ * pow(sin(a_ * pi_ * x), 0.2e1) *
      pow(cos(a_ * pi_ * y), 0.2e1) * pow(sin(a_ * pi_ * z), 0.2e1) * hnot_ *
      cos(ah_ * pi_ * x) * sin(ah_ * pi_ * y) * ah_ * pi_ * cos(ah_ * pi_ * z) /
      Cp_ -
    0.2e1 * Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      vnot_ * vnot_ * pow(sin(a_ * pi_ * x), 0.2e1) * cos(a_ * pi_ * y) *
      pow(sin(a_ * pi_ * z), 0.2e1) * sin(a_ * pi_ * y) * a_ * pi_ -
    Pref_ * MW_ / R_ *
      pow(
        hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
            Cp_ +
          Tref_,
        -0.2e1) *
      wnot_ * pow(sin(a_ * pi_ * x), 0.2e1) * sin(a_ * pi_ * y) *
      cos(a_ * pi_ * z) * vnot_ * cos(a_ * pi_ * y) * sin(a_ * pi_ * z) *
      hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * sin(ah_ * pi_ * z) *
      ah_ * pi_ / Cp_ +
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      wnot_ * pow(sin(a_ * pi_ * x), 0.2e1) * sin(a_ * pi_ * y) *
      pow(sin(a_ * pi_ * z), 0.2e1) * a_ * pi_ * vnot_ * cos(a_ * pi_ * y) -
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      wnot_ * pow(sin(a_ * pi_ * x), 0.2e1) * sin(a_ * pi_ * y) *
      pow(cos(a_ * pi_ * z), 0.2e1) * vnot_ * cos(a_ * pi_ * y) * a_ * pi_ -
    visc_ * (unot_ * sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
               cos(a_ * pi_ * y) * sin(a_ * pi_ * z) -
             vnot_ * sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
               cos(a_ * pi_ * y) * sin(a_ * pi_ * z)) -
    visc_ * (-(unot_ * sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
                 cos(a_ * pi_ * y) * sin(a_ * pi_ * z) -
               vnot_ * sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
                 cos(a_ * pi_ * y) * sin(a_ * pi_ * z) +
               wnot_ * sin(a_ * pi_ * x) * cos(a_ * pi_ * y) * a_ * a_ * pi_ *
                 pi_ * sin(a_ * pi_ * z)) *
               twoThirds_ -
             0.2e1 * vnot_ * sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
               cos(a_ * pi_ * y) * sin(a_ * pi_ * z)) -
    visc_ * (-vnot_ * sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
               cos(a_ * pi_ * y) * sin(a_ * pi_ * z) +
             wnot_ * sin(a_ * pi_ * x) * cos(a_ * pi_ * y) * a_ * a_ * pi_ *
               pi_ * sin(a_ * pi_ * z)) +
    0.50e0 * pnot_ * sin(0.2e1 * a_ * pi_ * y) * a_ * pi_ -
    (Pref_ * MW_ / R_ /
       (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
          Cp_ +
        Tref_) -
     rhoRef_) *
      gy_;

  srcXi_[2] =
    Pref_ * MW_ / R_ *
      pow(
        hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
            Cp_ +
          Tref_,
        -0.2e1) *
      wnot_ * sin(a_ * pi_ * x) * pow(sin(a_ * pi_ * y), 0.2e1) *
      cos(a_ * pi_ * z) * unot_ * cos(a_ * pi_ * x) * sin(a_ * pi_ * z) *
      hnot_ * sin(ah_ * pi_ * x) * ah_ * pi_ * cos(ah_ * pi_ * y) *
      cos(ah_ * pi_ * z) / Cp_ +
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      wnot_ * pow(cos(a_ * pi_ * x), 0.2e1) * a_ * pi_ *
      pow(sin(a_ * pi_ * y), 0.2e1) * cos(a_ * pi_ * z) * unot_ *
      sin(a_ * pi_ * z) -
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      wnot_ * pow(sin(a_ * pi_ * x), 0.2e1) * pow(sin(a_ * pi_ * y), 0.2e1) *
      cos(a_ * pi_ * z) * unot_ * a_ * pi_ * sin(a_ * pi_ * z) -
    Pref_ * MW_ / R_ *
      pow(
        hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
            Cp_ +
          Tref_,
        -0.2e1) *
      wnot_ * pow(sin(a_ * pi_ * x), 0.2e1) * sin(a_ * pi_ * y) *
      cos(a_ * pi_ * z) * vnot_ * cos(a_ * pi_ * y) * sin(a_ * pi_ * z) *
      hnot_ * cos(ah_ * pi_ * x) * sin(ah_ * pi_ * y) * ah_ * pi_ *
      cos(ah_ * pi_ * z) / Cp_ -
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      wnot_ * pow(sin(a_ * pi_ * x), 0.2e1) * pow(cos(a_ * pi_ * y), 0.2e1) *
      a_ * pi_ * cos(a_ * pi_ * z) * vnot_ * sin(a_ * pi_ * z) +
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      wnot_ * pow(sin(a_ * pi_ * x), 0.2e1) * pow(sin(a_ * pi_ * y), 0.2e1) *
      cos(a_ * pi_ * z) * vnot_ * a_ * pi_ * sin(a_ * pi_ * z) +
    Pref_ * MW_ / R_ *
      pow(
        hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
            Cp_ +
          Tref_,
        -0.2e1) *
      wnot_ * wnot_ * pow(sin(a_ * pi_ * x), 0.2e1) *
      pow(sin(a_ * pi_ * y), 0.2e1) * pow(cos(a_ * pi_ * z), 0.2e1) * hnot_ *
      cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * sin(ah_ * pi_ * z) * ah_ * pi_ /
      Cp_ -
    0.2e1 * Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      wnot_ * wnot_ * pow(sin(a_ * pi_ * x), 0.2e1) *
      pow(sin(a_ * pi_ * y), 0.2e1) * cos(a_ * pi_ * z) * sin(a_ * pi_ * z) *
      a_ * pi_ -
    visc_ * (unot_ * sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
               sin(a_ * pi_ * y) * cos(a_ * pi_ * z) +
             wnot_ * sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
               sin(a_ * pi_ * y) * cos(a_ * pi_ * z)) -
    visc_ * (-vnot_ * sin(a_ * pi_ * x) * sin(a_ * pi_ * y) * a_ * a_ * pi_ *
               pi_ * cos(a_ * pi_ * z) +
             wnot_ * sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
               sin(a_ * pi_ * y) * cos(a_ * pi_ * z)) -
    visc_ * (-(unot_ * sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
                 sin(a_ * pi_ * y) * cos(a_ * pi_ * z) -
               vnot_ * sin(a_ * pi_ * x) * sin(a_ * pi_ * y) * a_ * a_ * pi_ *
                 pi_ * cos(a_ * pi_ * z) +
               wnot_ * sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
                 sin(a_ * pi_ * y) * cos(a_ * pi_ * z)) *
               twoThirds_ +
             0.2e1 * wnot_ * sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ *
               sin(a_ * pi_ * y) * cos(a_ * pi_ * z)) +
    0.50e0 * pnot_ * sin(0.2e1 * a_ * pi_ * z) * a_ * pi_ -
    (Pref_ * MW_ / R_ /
       (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
          Cp_ +
        Tref_) -
     rhoRef_) *
      gz_;

  for (int i = 0; i < nDim_; ++i)
    rhs[i] += srcXi_[i] * dualVolume;
}

} // namespace nalu
} // namespace sierra
