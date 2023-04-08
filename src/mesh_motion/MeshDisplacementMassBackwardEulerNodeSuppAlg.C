// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <mesh_motion/MeshDisplacementMassBackwardEulerNodeSuppAlg.h>
#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <TimeIntegrator.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// MeshDisplacementMassBackwardEulerNodeSuppAlg - lumped mass BDF2
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MeshDisplacementMassBackwardEulerNodeSuppAlg::
  MeshDisplacementMassBackwardEulerNodeSuppAlg(Realm& realm)
  : SupplementalAlgorithm(realm),
    displacementN_(NULL),
    displacementNp1_(NULL),
    density_(NULL),
    dualNodalVolume_(NULL),
    dt_(0.0),
    nDim_(1)
{
  // save off fields
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  VectorFieldType* displacement = meta_data.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_displacement");
  displacementNp1_ = &(displacement->field_of_state(stk::mesh::StateNP1));
  displacementN_ = &(displacement->field_of_state(stk::mesh::StateN));
  density_ =
    meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");
  nDim_ = meta_data.spatial_dimension();
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
MeshDisplacementMassBackwardEulerNodeSuppAlg::setup()
{
  dt_ = realm_.timeIntegrator_->get_time_step();
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
MeshDisplacementMassBackwardEulerNodeSuppAlg::node_execute(
  double* lhs, double* rhs, stk::mesh::Entity node)
{
  // deal with lumped mass matrix (diagonal matrix)
  const double* dxN = stk::mesh::field_data(*displacementN_, node);
  const double* dxNp1 = stk::mesh::field_data(*displacementNp1_, node);
  const double rho = *stk::mesh::field_data(*density_, node);
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node);

  const double lhsfac = rho * dualVolume / dt_ / dt_;
  const int nDim = nDim_;
  for (int i = 0; i < nDim; ++i) {
    rhs[i] += -lhsfac * (dxNp1[i] - dxN[i]);
    const int row = i * nDim;
    lhs[row + i] += lhsfac;
  }
}

} // namespace nalu
} // namespace sierra
