// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/VariableDensityMomentumSrcNodeSuppAlg.h>
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
// VariableDensityMomentumSrcNodeSuppAlg - base class for algorithm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
VariableDensityMomentumSrcNodeSuppAlg::VariableDensityMomentumSrcNodeSuppAlg(
  Realm& realm)
  : SupplementalAlgorithm(realm),
    coordinates_(NULL),
    dualNodalVolume_(NULL),
    nDim_(realm_.spatialDimension_),
    unot_(1.0),
    vnot_(1.0),
    wnot_(1.0),
    pnot_(1.0),
    znot_(1.0),
    a_(20.0),
    amf_(10.0),
    visc_(0.001),
    rhoP_(0.1),
    rhoS_(1.0),
    pi_(std::acos(-1.0)),
    twoThirds_(2.0 / 3.0 * realm_.get_divU()),
    rhoRef_(1.0),
    gx_(0.0),
    gy_(0.0),
    gz_(0.0)
{
  // save off fields
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name());
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");

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
VariableDensityMomentumSrcNodeSuppAlg::setup()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
VariableDensityMomentumSrcNodeSuppAlg::node_execute(
  double* /*lhs*/, double* rhs, stk::mesh::Entity node)
{
  // deal with lumped mass matrix
  const double* coords = stk::mesh::field_data(*coordinates_, node);
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node);
  const double x = coords[0];
  const double y = coords[1];
  const double z = coords[2];

  srcXi_[0] =
    -0.10e1 *
      pow(
        znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
            cos(amf_ * pi_ * z) / rhoP_ +
          (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                     cos(amf_ * pi_ * z)) /
            rhoS_,
        -0.2e1) *
      unot_ * unot_ * pow(cos(a_ * pi_ * x), 0.2e1) *
      pow(sin(a_ * pi_ * y), 0.2e1) * pow(sin(a_ * pi_ * z), 0.2e1) *
      (-znot_ * sin(amf_ * pi_ * x) * amf_ * pi_ * cos(amf_ * pi_ * y) *
         cos(amf_ * pi_ * z) / rhoP_ +
       znot_ * sin(amf_ * pi_ * x) * amf_ * pi_ * cos(amf_ * pi_ * y) *
         cos(amf_ * pi_ * z) / rhoS_) -
    0.20e1 /
      (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * cos(amf_ * pi_ * z) /
         rhoP_ +
       (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                  cos(amf_ * pi_ * z)) /
         rhoS_) *
      unot_ * unot_ * cos(a_ * pi_ * x) * pow(sin(a_ * pi_ * y), 0.2e1) *
      pow(sin(a_ * pi_ * z), 0.2e1) * sin(a_ * pi_ * x) * a_ * pi_ +
    0.10e1 *
      pow(
        znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
            cos(amf_ * pi_ * z) / rhoP_ +
          (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                     cos(amf_ * pi_ * z)) /
            rhoS_,
        -0.2e1) *
      vnot_ * sin(a_ * pi_ * x) * cos(a_ * pi_ * y) *
      pow(sin(a_ * pi_ * z), 0.2e1) * unot_ * cos(a_ * pi_ * x) *
      sin(a_ * pi_ * y) *
      (-znot_ * cos(amf_ * pi_ * x) * sin(amf_ * pi_ * y) * amf_ * pi_ *
         cos(amf_ * pi_ * z) / rhoP_ +
       znot_ * cos(amf_ * pi_ * x) * sin(amf_ * pi_ * y) * amf_ * pi_ *
         cos(amf_ * pi_ * z) / rhoS_) +
    0.10e1 /
      (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * cos(amf_ * pi_ * z) /
         rhoP_ +
       (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                  cos(amf_ * pi_ * z)) /
         rhoS_) *
      vnot_ * sin(a_ * pi_ * x) * pow(sin(a_ * pi_ * y), 0.2e1) * a_ * pi_ *
      pow(sin(a_ * pi_ * z), 0.2e1) * unot_ * cos(a_ * pi_ * x) -
    0.10e1 /
      (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * cos(amf_ * pi_ * z) /
         rhoP_ +
       (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                  cos(amf_ * pi_ * z)) /
         rhoS_) *
      vnot_ * sin(a_ * pi_ * x) * pow(cos(a_ * pi_ * y), 0.2e1) *
      pow(sin(a_ * pi_ * z), 0.2e1) * unot_ * cos(a_ * pi_ * x) * a_ * pi_ -
    0.10e1 *
      pow(
        znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
            cos(amf_ * pi_ * z) / rhoP_ +
          (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                     cos(amf_ * pi_ * z)) /
            rhoS_,
        -0.2e1) *
      wnot_ * sin(a_ * pi_ * x) * pow(sin(a_ * pi_ * y), 0.2e1) *
      cos(a_ * pi_ * z) * unot_ * cos(a_ * pi_ * x) * sin(a_ * pi_ * z) *
      (-znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
         sin(amf_ * pi_ * z) * amf_ * pi_ / rhoP_ +
       znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * sin(amf_ * pi_ * z) *
         amf_ * pi_ / rhoS_) -
    0.10e1 /
      (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * cos(amf_ * pi_ * z) /
         rhoP_ +
       (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                  cos(amf_ * pi_ * z)) /
         rhoS_) *
      wnot_ * sin(a_ * pi_ * x) * pow(sin(a_ * pi_ * y), 0.2e1) *
      pow(sin(a_ * pi_ * z), 0.2e1) * a_ * pi_ * unot_ * cos(a_ * pi_ * x) +
    0.10e1 /
      (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * cos(amf_ * pi_ * z) /
         rhoP_ +
       (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                  cos(amf_ * pi_ * z)) /
         rhoS_) *
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
    (0.10e1 / (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                 cos(amf_ * pi_ * z) / rhoP_ +
               (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                          cos(amf_ * pi_ * z)) /
                 rhoS_) -
     rhoRef_) *
      gx_;

  srcXi_[1] =
    0.10e1 *
      pow(
        znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
            cos(amf_ * pi_ * z) / rhoP_ +
          (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                     cos(amf_ * pi_ * z)) /
            rhoS_,
        -0.2e1) *
      vnot_ * sin(a_ * pi_ * x) * cos(a_ * pi_ * y) *
      pow(sin(a_ * pi_ * z), 0.2e1) * unot_ * cos(a_ * pi_ * x) *
      sin(a_ * pi_ * y) *
      (-znot_ * sin(amf_ * pi_ * x) * amf_ * pi_ * cos(amf_ * pi_ * y) *
         cos(amf_ * pi_ * z) / rhoP_ +
       znot_ * sin(amf_ * pi_ * x) * amf_ * pi_ * cos(amf_ * pi_ * y) *
         cos(amf_ * pi_ * z) / rhoS_) -
    0.10e1 /
      (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * cos(amf_ * pi_ * z) /
         rhoP_ +
       (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                  cos(amf_ * pi_ * z)) /
         rhoS_) *
      vnot_ * pow(cos(a_ * pi_ * x), 0.2e1) * a_ * pi_ * cos(a_ * pi_ * y) *
      pow(sin(a_ * pi_ * z), 0.2e1) * unot_ * sin(a_ * pi_ * y) +
    0.10e1 /
      (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * cos(amf_ * pi_ * z) /
         rhoP_ +
       (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                  cos(amf_ * pi_ * z)) /
         rhoS_) *
      vnot_ * pow(sin(a_ * pi_ * x), 0.2e1) * cos(a_ * pi_ * y) *
      pow(sin(a_ * pi_ * z), 0.2e1) * unot_ * a_ * pi_ * sin(a_ * pi_ * y) -
    0.10e1 *
      pow(
        znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
            cos(amf_ * pi_ * z) / rhoP_ +
          (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                     cos(amf_ * pi_ * z)) /
            rhoS_,
        -0.2e1) *
      vnot_ * vnot_ * pow(sin(a_ * pi_ * x), 0.2e1) *
      pow(cos(a_ * pi_ * y), 0.2e1) * pow(sin(a_ * pi_ * z), 0.2e1) *
      (-znot_ * cos(amf_ * pi_ * x) * sin(amf_ * pi_ * y) * amf_ * pi_ *
         cos(amf_ * pi_ * z) / rhoP_ +
       znot_ * cos(amf_ * pi_ * x) * sin(amf_ * pi_ * y) * amf_ * pi_ *
         cos(amf_ * pi_ * z) / rhoS_) -
    0.20e1 /
      (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * cos(amf_ * pi_ * z) /
         rhoP_ +
       (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                  cos(amf_ * pi_ * z)) /
         rhoS_) *
      vnot_ * vnot_ * pow(sin(a_ * pi_ * x), 0.2e1) * cos(a_ * pi_ * y) *
      pow(sin(a_ * pi_ * z), 0.2e1) * sin(a_ * pi_ * y) * a_ * pi_ +
    0.10e1 *
      pow(
        znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
            cos(amf_ * pi_ * z) / rhoP_ +
          (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                     cos(amf_ * pi_ * z)) /
            rhoS_,
        -0.2e1) *
      wnot_ * pow(sin(a_ * pi_ * x), 0.2e1) * sin(a_ * pi_ * y) *
      cos(a_ * pi_ * z) * vnot_ * cos(a_ * pi_ * y) * sin(a_ * pi_ * z) *
      (-znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
         sin(amf_ * pi_ * z) * amf_ * pi_ / rhoP_ +
       znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * sin(amf_ * pi_ * z) *
         amf_ * pi_ / rhoS_) +
    0.10e1 /
      (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * cos(amf_ * pi_ * z) /
         rhoP_ +
       (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                  cos(amf_ * pi_ * z)) /
         rhoS_) *
      wnot_ * pow(sin(a_ * pi_ * x), 0.2e1) * sin(a_ * pi_ * y) *
      pow(sin(a_ * pi_ * z), 0.2e1) * a_ * pi_ * vnot_ * cos(a_ * pi_ * y) -
    0.10e1 /
      (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * cos(amf_ * pi_ * z) /
         rhoP_ +
       (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                  cos(amf_ * pi_ * z)) /
         rhoS_) *
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
    (0.10e1 / (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                 cos(amf_ * pi_ * z) / rhoP_ +
               (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                          cos(amf_ * pi_ * z)) /
                 rhoS_) -
     rhoRef_) *
      gy_;

  srcXi_[2] =
    -0.10e1 *
      pow(
        znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
            cos(amf_ * pi_ * z) / rhoP_ +
          (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                     cos(amf_ * pi_ * z)) /
            rhoS_,
        -0.2e1) *
      wnot_ * sin(a_ * pi_ * x) * pow(sin(a_ * pi_ * y), 0.2e1) *
      cos(a_ * pi_ * z) * unot_ * cos(a_ * pi_ * x) * sin(a_ * pi_ * z) *
      (-znot_ * sin(amf_ * pi_ * x) * amf_ * pi_ * cos(amf_ * pi_ * y) *
         cos(amf_ * pi_ * z) / rhoP_ +
       znot_ * sin(amf_ * pi_ * x) * amf_ * pi_ * cos(amf_ * pi_ * y) *
         cos(amf_ * pi_ * z) / rhoS_) +
    0.10e1 /
      (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * cos(amf_ * pi_ * z) /
         rhoP_ +
       (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                  cos(amf_ * pi_ * z)) /
         rhoS_) *
      wnot_ * pow(cos(a_ * pi_ * x), 0.2e1) * a_ * pi_ *
      pow(sin(a_ * pi_ * y), 0.2e1) * cos(a_ * pi_ * z) * unot_ *
      sin(a_ * pi_ * z) -
    0.10e1 /
      (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * cos(amf_ * pi_ * z) /
         rhoP_ +
       (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                  cos(amf_ * pi_ * z)) /
         rhoS_) *
      wnot_ * pow(sin(a_ * pi_ * x), 0.2e1) * pow(sin(a_ * pi_ * y), 0.2e1) *
      cos(a_ * pi_ * z) * unot_ * a_ * pi_ * sin(a_ * pi_ * z) +
    0.10e1 *
      pow(
        znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
            cos(amf_ * pi_ * z) / rhoP_ +
          (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                     cos(amf_ * pi_ * z)) /
            rhoS_,
        -0.2e1) *
      wnot_ * pow(sin(a_ * pi_ * x), 0.2e1) * sin(a_ * pi_ * y) *
      cos(a_ * pi_ * z) * vnot_ * cos(a_ * pi_ * y) * sin(a_ * pi_ * z) *
      (-znot_ * cos(amf_ * pi_ * x) * sin(amf_ * pi_ * y) * amf_ * pi_ *
         cos(amf_ * pi_ * z) / rhoP_ +
       znot_ * cos(amf_ * pi_ * x) * sin(amf_ * pi_ * y) * amf_ * pi_ *
         cos(amf_ * pi_ * z) / rhoS_) -
    0.10e1 /
      (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * cos(amf_ * pi_ * z) /
         rhoP_ +
       (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                  cos(amf_ * pi_ * z)) /
         rhoS_) *
      wnot_ * pow(sin(a_ * pi_ * x), 0.2e1) * pow(cos(a_ * pi_ * y), 0.2e1) *
      a_ * pi_ * cos(a_ * pi_ * z) * vnot_ * sin(a_ * pi_ * z) +
    0.10e1 /
      (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * cos(amf_ * pi_ * z) /
         rhoP_ +
       (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                  cos(amf_ * pi_ * z)) /
         rhoS_) *
      wnot_ * pow(sin(a_ * pi_ * x), 0.2e1) * pow(sin(a_ * pi_ * y), 0.2e1) *
      cos(a_ * pi_ * z) * vnot_ * a_ * pi_ * sin(a_ * pi_ * z) -
    0.10e1 *
      pow(
        znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
            cos(amf_ * pi_ * z) / rhoP_ +
          (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                     cos(amf_ * pi_ * z)) /
            rhoS_,
        -0.2e1) *
      wnot_ * wnot_ * pow(sin(a_ * pi_ * x), 0.2e1) *
      pow(sin(a_ * pi_ * y), 0.2e1) * pow(cos(a_ * pi_ * z), 0.2e1) *
      (-znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
         sin(amf_ * pi_ * z) * amf_ * pi_ / rhoP_ +
       znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * sin(amf_ * pi_ * z) *
         amf_ * pi_ / rhoS_) -
    0.20e1 /
      (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) * cos(amf_ * pi_ * z) /
         rhoP_ +
       (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                  cos(amf_ * pi_ * z)) /
         rhoS_) *
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
    (0.10e1 / (znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                 cos(amf_ * pi_ * z) / rhoP_ +
               (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                          cos(amf_ * pi_ * z)) /
                 rhoS_) -
     rhoRef_) *
      gz_;

  for (int i = 0; i < nDim_; ++i)
    rhs[i] += srcXi_[i] * dualVolume;
}

} // namespace nalu
} // namespace sierra
