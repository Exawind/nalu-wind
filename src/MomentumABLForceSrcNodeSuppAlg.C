
#include "MomentumABLForceSrcNodeSuppAlg.h"
#include "Realm.h"
#include "wind_energy/ABLForcingAlgorithm.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

MomentumABLForceSrcNodeSuppAlg::MomentumABLForceSrcNodeSuppAlg(
  Realm& realm, ABLForcingAlgorithm* ablsrc)
  : SupplementalAlgorithm(realm),
    ablSrc_(ablsrc),
    coords_(realm.meta_data().get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "coordinates")),
    dualNodalVolume_(realm.meta_data().get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "dual_nodal_volume")),
    nDim_(realm_.meta_data().spatial_dimension())
  {}

void
MomentumABLForceSrcNodeSuppAlg::node_execute(
  double*  /* lhs */, double* rhs, stk::mesh::Entity node)
{
  const double dualVol = *stk::mesh::field_data(*dualNodalVolume_, node);
  const double* pt = stk::mesh::field_data(*coords_, node);
  //~ const double rhoNP1 = *stk::mesh::field_data(*density_, node);
  std::vector<double> momSrc(nDim_);

  // Evaluate the momentum source
  ablSrc_->eval_momentum_source(pt[nDim_ - 1], momSrc);

  // Add the momentum source into the RHS
  for (int i = 0; i < nDim_; i++) {
    rhs[i] += dualVol * momSrc[i];
  }
}

} // namespace nalu
} // namespace sierra
