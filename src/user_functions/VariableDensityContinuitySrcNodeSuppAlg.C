// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/VariableDensityContinuitySrcNodeSuppAlg.h>
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
// VariableDensityContinuitySrcNodeSuppAlg - base class for algorithm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
VariableDensityContinuitySrcNodeSuppAlg::
  VariableDensityContinuitySrcNodeSuppAlg(Realm& realm)
  : SupplementalAlgorithm(realm),
    coordinates_(NULL),
    dualNodalVolume_(NULL),
    unot_(1.0),
    vnot_(1.0),
    wnot_(1.0),
    znot_(1.0),
    rhoP_(0.1),
    rhoS_(1.0),
    a_(20.0),
    amf_(10.0),
    pi_(std::acos(-1.0)),
    projTimeScale_(1.0)
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
VariableDensityContinuitySrcNodeSuppAlg::setup()
{
  projTimeScale_ = realm_.get_time_step() / realm_.get_gamma1();
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
VariableDensityContinuitySrcNodeSuppAlg::node_execute(
  double* /*lhs*/, double* rhs, stk::mesh::Entity node)
{
  // deal with lumped mass matrix
  const double* coords = stk::mesh::field_data(*coordinates_, node);
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node);
  const double x = coords[0];
  const double y = coords[1];
  const double z = coords[2];

  const double src =
    0.10e1 *
      pow(
        znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
            cos(amf_ * pi_ * z) / rhoP_ +
          (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                     cos(amf_ * pi_ * z)) /
            rhoS_,
        -0.2e1) *
      unot_ * cos(a_ * pi_ * x) * sin(a_ * pi_ * y) * sin(a_ * pi_ * z) *
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
      unot_ * sin(a_ * pi_ * x) * a_ * pi_ * sin(a_ * pi_ * y) *
      sin(a_ * pi_ * z) -
    0.10e1 *
      pow(
        znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
            cos(amf_ * pi_ * z) / rhoP_ +
          (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                     cos(amf_ * pi_ * z)) /
            rhoS_,
        -0.2e1) *
      vnot_ * sin(a_ * pi_ * x) * cos(a_ * pi_ * y) * sin(a_ * pi_ * z) *
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
      vnot_ * sin(a_ * pi_ * x) * sin(a_ * pi_ * y) * a_ * pi_ *
      sin(a_ * pi_ * z) +
    0.10e1 *
      pow(
        znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
            cos(amf_ * pi_ * z) / rhoP_ +
          (0.1e1 - znot_ * cos(amf_ * pi_ * x) * cos(amf_ * pi_ * y) *
                     cos(amf_ * pi_ * z)) /
            rhoS_,
        -0.2e1) *
      wnot_ * sin(a_ * pi_ * x) * sin(a_ * pi_ * y) * cos(a_ * pi_ * z) *
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
      wnot_ * sin(a_ * pi_ * x) * sin(a_ * pi_ * y) * sin(a_ * pi_ * z) * a_ *
      pi_;

  rhs[0] += src * dualVolume / projTimeScale_;
}

} // namespace nalu
} // namespace sierra
