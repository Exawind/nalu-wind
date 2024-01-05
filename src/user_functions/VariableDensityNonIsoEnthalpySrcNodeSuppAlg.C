// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/VariableDensityNonIsoEnthalpySrcNodeSuppAlg.h>
#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <TimeIntegrator.h>

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
// VariableDensityNonIsoEnthalpySrcNodeSuppAlg - base class for algorithm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
VariableDensityNonIsoEnthalpySrcNodeSuppAlg::
  VariableDensityNonIsoEnthalpySrcNodeSuppAlg(Realm& realm)
  : SupplementalAlgorithm(realm),
    coordinates_(NULL),
    dualNodalVolume_(NULL),
    unot_(1.0),
    vnot_(1.0),
    wnot_(1.0),
    hnot_(1.0),
    a_(20.0),
    ah_(10.0),
    visc_(0.00125),
    Pref_(100.0),
    MW_(30.0),
    R_(10.0),
    Tref_(300.0),
    Cp_(0.01),
    Pr_(0.8),
    pi_(acos(-1.0))
{
  // save off fields
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name());
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
VariableDensityNonIsoEnthalpySrcNodeSuppAlg::setup()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
VariableDensityNonIsoEnthalpySrcNodeSuppAlg::node_execute(
  double* /*lhs*/, double* rhs, stk::mesh::Entity node)
{
  // deal with lumped mass matrix
  const double* coords = stk::mesh::field_data(*coordinates_, node);
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node);

  const double x = coords[0];
  const double y = coords[1];
  const double z = coords[2];

  const double src =
    -Pref_ * MW_ / R_ *
      pow(
        hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
            Cp_ +
          Tref_,
        -0.2e1) *
      unot_ * cos(a_ * pi_ * x) * sin(a_ * pi_ * y) * sin(a_ * pi_ * z) *
      hnot_ * hnot_ * cos(ah_ * pi_ * x) * pow(cos(ah_ * pi_ * y), 0.2e1) *
      pow(cos(ah_ * pi_ * z), 0.2e1) * sin(ah_ * pi_ * x) * ah_ * pi_ / Cp_ +
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      unot_ * sin(a_ * pi_ * x) * a_ * pi_ * sin(a_ * pi_ * y) *
      sin(a_ * pi_ * z) * hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) *
      cos(ah_ * pi_ * z) +
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      unot_ * cos(a_ * pi_ * x) * sin(a_ * pi_ * y) * sin(a_ * pi_ * z) *
      hnot_ * sin(ah_ * pi_ * x) * ah_ * pi_ * cos(ah_ * pi_ * y) *
      cos(ah_ * pi_ * z) +
    Pref_ * MW_ / R_ *
      pow(
        hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
            Cp_ +
          Tref_,
        -0.2e1) *
      vnot_ * sin(a_ * pi_ * x) * cos(a_ * pi_ * y) * sin(a_ * pi_ * z) *
      hnot_ * hnot_ * pow(cos(ah_ * pi_ * x), 0.2e1) * cos(ah_ * pi_ * y) *
      pow(cos(ah_ * pi_ * z), 0.2e1) * sin(ah_ * pi_ * y) * ah_ * pi_ / Cp_ -
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      vnot_ * sin(a_ * pi_ * x) * sin(a_ * pi_ * y) * a_ * pi_ *
      sin(a_ * pi_ * z) * hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) *
      cos(ah_ * pi_ * z) -
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      vnot_ * sin(a_ * pi_ * x) * cos(a_ * pi_ * y) * sin(a_ * pi_ * z) *
      hnot_ * cos(ah_ * pi_ * x) * sin(ah_ * pi_ * y) * ah_ * pi_ *
      cos(ah_ * pi_ * z) -
    Pref_ * MW_ / R_ *
      pow(
        hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
            Cp_ +
          Tref_,
        -0.2e1) *
      wnot_ * sin(a_ * pi_ * x) * sin(a_ * pi_ * y) * cos(a_ * pi_ * z) *
      hnot_ * hnot_ * pow(cos(ah_ * pi_ * x), 0.2e1) *
      pow(cos(ah_ * pi_ * y), 0.2e1) * cos(ah_ * pi_ * z) * sin(ah_ * pi_ * z) *
      ah_ * pi_ / Cp_ +
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      wnot_ * sin(a_ * pi_ * x) * sin(a_ * pi_ * y) * sin(a_ * pi_ * z) * a_ *
      pi_ * hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) *
      cos(ah_ * pi_ * z) +
    Pref_ * MW_ / R_ /
      (hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z) /
         Cp_ +
       Tref_) *
      wnot_ * sin(a_ * pi_ * x) * sin(a_ * pi_ * y) * cos(a_ * pi_ * z) *
      hnot_ * cos(ah_ * pi_ * x) * cos(ah_ * pi_ * y) * sin(ah_ * pi_ * z) *
      ah_ * pi_ +
    0.3e1 * visc_ / Pr_ * hnot_ * cos(ah_ * pi_ * x) * ah_ * ah_ * pi_ * pi_ *
      cos(ah_ * pi_ * y) * cos(ah_ * pi_ * z);

  rhs[0] += src * dualVolume;
}

} // namespace nalu
} // namespace sierra
