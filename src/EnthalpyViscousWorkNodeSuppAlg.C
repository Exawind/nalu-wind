// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <EnthalpyViscousWorkNodeSuppAlg.h>
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
// EnthalpyViscousWorkNodeSuppAlg - base class for algorithm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
EnthalpyViscousWorkNodeSuppAlg::EnthalpyViscousWorkNodeSuppAlg(Realm& realm)
  : SupplementalAlgorithm(realm),
    dudx_(NULL),
    viscosity_(NULL),
    dualNodalVolume_(NULL),
    includeDivU_(realm_.get_divU()),
    nDim_(realm_.meta_data().spatial_dimension())
{
  // save off fields
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  dudx_ =
    meta_data.get_field<TensorFieldType>(stk::topology::NODE_RANK, "dudx");
  const std::string viscName =
    realm.is_turbulent() ? "effective_viscosity_u" : "viscosity";
  viscosity_ =
    meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, viscName);
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyViscousWorkNodeSuppAlg::setup()
{
  // nothing to do now
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyViscousWorkNodeSuppAlg::node_execute(
  double* lhs, double* rhs, stk::mesh::Entity node)
{
  // viscous work
  const double* dudx = stk::mesh::field_data(*dudx_, node);
  const double viscosity = *stk::mesh::field_data(*viscosity_, node);
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node);

  // form divU
  double divU = 0.0;
  for (int j = 0; j < nDim_; ++j) {
    const int row = j * nDim_;
    divU += dudx[row + j];
  }

  double viscousWork = 0.0;
  for (int i = 0; i < nDim_; ++i) {
    const int offSet = nDim_ * i;
    for (int j = 0; j < nDim_; ++j) {
      viscousWork +=
        dudx[offSet + j] * (dudx[offSet + j] + dudx[nDim_ * j + i]);
      if (i == j)
        viscousWork -= dudx[offSet + j] * 2.0 / 3.0 * divU * includeDivU_;
    }
  }
  viscousWork *= viscosity;

  // assemble
  rhs[0] += viscousWork * dualVolume;
  lhs[0] += 0.0;
}

} // namespace nalu
} // namespace sierra
