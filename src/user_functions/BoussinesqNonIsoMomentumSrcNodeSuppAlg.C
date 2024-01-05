// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/BoussinesqNonIsoMomentumSrcNodeSuppAlg.h>
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
// BoussinesqNonIsoMomentumSrcNodeSuppAlg - base class for algorithm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
BoussinesqNonIsoMomentumSrcNodeSuppAlg::BoussinesqNonIsoMomentumSrcNodeSuppAlg(
  Realm& realm)
  : SupplementalAlgorithm(realm),
    coordinates_(NULL),
    dualNodalVolume_(NULL),
    visc_(0.00125),
    Cp(0.01),
    rhoRef(1.0)
{
  // save off fields
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name());
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");

  // extract user parameters from solution options
  gravity = realm_.solutionOptions_->gravity_;
  rhoRef = realm_.solutionOptions_->referenceDensity_;
  TRef = realm_.solutionOptions_->referenceTemperature_;
  beta = realm_.solutionOptions_->thermalExpansionCoeff_;
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
BoussinesqNonIsoMomentumSrcNodeSuppAlg::setup()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
BoussinesqNonIsoMomentumSrcNodeSuppAlg::node_execute(
  double* /*lhs*/, double* rhs, stk::mesh::Entity node)
{
  // deal with lumped mass matrix
  const double* coords = stk::mesh::field_data(*coordinates_, node);
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node);
  const double mu = visc_;

  const double x = coords[0];
  const double y = coords[1];
  const double z = coords[2];

  double srcXi[3] = {0, 0, 0};
  srcXi[0] = -(M_PI * (1 + cos(4 * M_PI * y) - 2 * cos(4 * M_PI * z)) *
               sin(4 * M_PI * x)) /
               8. +
             6 * mu * (M_PI * M_PI) * cos(2 * M_PI * x) * sin(2 * M_PI * y) *
               sin(2 * M_PI * z);
  srcXi[1] =
    (M_PI * ((-2 + cos(4 * M_PI * x) + cos(4 * M_PI * z)) * sin(4 * M_PI * y) -
             48 * mu * M_PI * cos(2 * M_PI * y) * sin(2 * M_PI * x) *
               sin(2 * M_PI * z))) /
    4.;
  srcXi[2] =
    (M_PI * (24 * mu * M_PI * cos(2 * M_PI * z) * sin(2 * M_PI * x) *
               sin(2 * M_PI * y) +
             (cos(4 * M_PI * x) - (cos(2 * M_PI * y) * cos(2 * M_PI * y))) *
               sin(4 * M_PI * z))) /
    4.;

  for (int d = 0; d < 3; ++d) {
    rhs[d] += srcXi[d] * dualVolume;
  }

  const double h = z;
  const double temperature = h / Cp + TRef;

  const double fac = rhoRef * beta * (temperature - TRef) * dualVolume;
  for (int d = 0; d < 3; ++d) {
    rhs[d] += fac * gravity[d];
  }
}

} // namespace nalu
} // namespace sierra
