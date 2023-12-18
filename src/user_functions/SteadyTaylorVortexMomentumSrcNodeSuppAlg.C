// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/SteadyTaylorVortexMomentumSrcNodeSuppAlg.h>
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
// SteadyTaylorVortexMomentumSrcNodeSuppAlg - base class for algorithm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
SteadyTaylorVortexMomentumSrcNodeSuppAlg::
  SteadyTaylorVortexMomentumSrcNodeSuppAlg(Realm& realm)
  : SupplementalAlgorithm(realm),
    coordinates_(NULL),
    dualNodalVolume_(NULL),
    nDim_(realm_.spatialDimension_),
    unot_(1.0),
    a_(20.0),
    visc_(0.001),
    pi_(std::acos(-1.0))
{
  // save off fields
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<double>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name());
  dualNodalVolume_ =
    meta_data.get_field<double>(stk::topology::NODE_RANK, "dual_nodal_volume");

  // internal source
  srcXi_.resize(nDim_);
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
SteadyTaylorVortexMomentumSrcNodeSuppAlg::setup()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
SteadyTaylorVortexMomentumSrcNodeSuppAlg::node_execute(
  double* /*lhs*/, double* rhs, stk::mesh::Entity node)
{
  // deal with lumped mass matrix
  const double* coords = stk::mesh::field_data(*coordinates_, node);
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node);
  const double x = coords[0];
  const double y = coords[1];

  srcXi_[0] = -2.0 * unot_ * a_ * a_ * pi_ * pi_ * visc_ * cos(a_ * pi_ * x) *
              sin(a_ * pi_ * y);
  srcXi_[1] = 2.0 * unot_ * a_ * a_ * pi_ * pi_ * visc_ * sin(a_ * pi_ * x) *
              cos(a_ * pi_ * y);

  for (int i = 0; i < nDim_; ++i)
    rhs[i] += srcXi_[i] * dualVolume;
}

} // namespace nalu
} // namespace sierra
