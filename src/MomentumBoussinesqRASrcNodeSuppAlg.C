// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <MomentumBoussinesqRASrcNodeSuppAlg.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <SolutionOptions.h>
#include <SupplementalAlgorithm.h>
#include <TimeIntegrator.h>
#include <MovingAveragePostProcessor.h>

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
// MomentumBoussinesqSrcNodeSuppAlg - -rho*beta*(T-Tref) g
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MomentumBoussinesqRASrcNodeSuppAlg::MomentumBoussinesqRASrcNodeSuppAlg(
  Realm& realm)
  : SupplementalAlgorithm(realm),
    temperature_(NULL),
    dualNodalVolume_(NULL),
    rhoRef_(1.0),
    beta_(1.0),
    nDim_(1)
{
  if (!realm_.solutionOptions_->has_set_boussinesq_time_scale()) {
    throw std::runtime_error(
      "User must specify a timescale for the averaged Boussinesq model");
  }

  // save off fields
  stk::mesh::MetaData& meta_data = realm_.meta_data();

  temperature_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "temperature");
  ThrowRequire(temperature_ != nullptr);

  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");
  ThrowRequire(dualNodalVolume_ != nullptr);

  rhoRef_ = realm_.solutionOptions_->referenceDensity_;
  beta_ = realm_.solutionOptions_->thermalExpansionCoeff_;

  nDim_ = meta_data.spatial_dimension();
  gravity_ = realm_.solutionOptions_->gravity_;
}
//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumBoussinesqRASrcNodeSuppAlg::setup()
{
  // filtered temperature is registered after this alg is created
  raTemperature_ = realm_.meta_data().get_field<ScalarFieldType>(
    stk::topology::NODE_RANK,
    MovingAveragePostProcessor::filtered_field_name("temperature"));
  ThrowRequire(raTemperature_ != nullptr);
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumBoussinesqRASrcNodeSuppAlg::node_execute(
  double* /*lhs*/, double* rhs, stk::mesh::Entity node)
{
  const double temperature = *stk::mesh::field_data(*temperature_, node);
  const double raTemperature = *stk::mesh::field_data(*raTemperature_, node);
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node);

  const double fac =
    -rhoRef_ * beta_ * (temperature - raTemperature) * dualVolume;
  for (int i = 0; i < nDim_; ++i) {
    rhs[i] += fac * gravity_[i];
  }
}

} // namespace nalu
} // namespace sierra
