/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <user_functions/TGMMSMomentumSrcNodeSuppAlg.h>
#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <SolutionOptions.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra{
namespace nalu{

TGMMSMomentumSrcNodeSuppAlg::TGMMSMomentumSrcNodeSuppAlg(Realm &realm)
  : SupplementalAlgorithm(realm)
{
  auto& meta = realm_.meta_data();
  coordinates_ = meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  dualNodalVolume_ = meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
}

void
TGMMSMomentumSrcNodeSuppAlg::node_execute(
  double */*lhs*/,
  double *rhs,
  stk::mesh::Entity node)
{
  const double* xc = stk::mesh::field_data(*coordinates_, node);
  const double x = xc[0];
  const double y = xc[1];
  const double z = xc[2];
  constexpr double mu{1.0e-3};


  const double dv = *stk::mesh::field_data(*dualNodalVolume_, node );
  rhs[0] += dv*(-(M_PI*(1 + std::cos(4*M_PI*y) - 2*std::cos(4*M_PI*z))*std::sin(4*M_PI*x))/8.
      + 6*mu*(M_PI * M_PI)*std::cos(2*M_PI*x)*std::sin(2*M_PI*y)*std::sin(2*M_PI*z));

  rhs[1] += dv*(M_PI*((-2 + std::cos(4*M_PI*x) + std::cos(4*M_PI*z))*std::sin(4*M_PI*y)
      - 48*mu*M_PI*std::cos(2*M_PI*y)*std::sin(2*M_PI*x)*std::sin(2*M_PI*z)))/4.;

  rhs[2] += dv*(M_PI*(24*mu*M_PI*std::cos(2*M_PI*z)*std::sin(2*M_PI*x)*std::sin(2*M_PI*y)
      + (std::cos(4*M_PI*x) - (std::cos(2*M_PI*y) * std::cos(2*M_PI*y)))*std::sin(4*M_PI*z)))/4.;
}

} // namespace nalu
} // namespace Sierra
