/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level KynemaUGF */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef BLTGAMMADLRNODEKERNEL_H
#define BLTGAMMADLRNODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "FieldTypeDef.h"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace kynema_ugf {

/*------------------------------------------------------------------------*/
/* BLTGammaDLRNodeKernel is a correlation-based transition model          */
/* consisting of the baseline k-omega SST plus one augmented transport    */
/* equation for Gamma (turbulence intermittency): DLR-gamma model         */
/*                                                                        */
/* Francois, D. et al., Simplified Stability-Based Transition Transport   */
/* Modeling for Unstructured Computational Fluid Dynamics, Journal of     */
/* Aircraft, Vol. 60, No. 6, Nov. 2023. https://doi.org/10.2514/1.C037163 */
/*------------------------------------------------------------------------*/

class Realm;

class BLTGammaDLRNodeKernel : public NGPNodeKernel<BLTGammaDLRNodeKernel>
{
public:
  BLTGammaDLRNodeKernel(
    const stk::mesh::MetaData&,
    const std::string& velocityName,
    const double uInf);

  BLTGammaDLRNodeKernel() = delete;

  virtual ~BLTGammaDLRNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> tke_;
  stk::mesh::NgpField<double> sdr_;
  stk::mesh::NgpField<double> density_;
  stk::mesh::NgpField<double> visc_;
  stk::mesh::NgpField<double> tvisc_;
  stk::mesh::NgpField<double> velocity_;
  stk::mesh::NgpField<double> pressure_;
  stk::mesh::NgpField<double> dpdx_;
  stk::mesh::NgpField<double> dudx_;
  stk::mesh::NgpField<double> minD_;
  stk::mesh::NgpField<double> dualNodalVolume_;
  stk::mesh::NgpField<double> gamint_;

  unsigned tkeID_{stk::mesh::InvalidOrdinal};
  unsigned sdrID_{stk::mesh::InvalidOrdinal};
  unsigned densityID_{stk::mesh::InvalidOrdinal};
  unsigned viscID_{stk::mesh::InvalidOrdinal};
  unsigned tviscID_{stk::mesh::InvalidOrdinal};
  unsigned velocityID_{stk::mesh::InvalidOrdinal};
  unsigned pressureID_{stk::mesh::InvalidOrdinal};
  unsigned dpdxID_{stk::mesh::InvalidOrdinal};
  unsigned dudxID_{stk::mesh::InvalidOrdinal};
  unsigned minDID_{stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolumeID_{stk::mesh::InvalidOrdinal};
  unsigned gamintID_{stk::mesh::InvalidOrdinal};

  NodeKernelTraits::DblType fsti_;
  NodeKernelTraits::DblType uInf_;

  const int nDim_;
};

} // namespace kynema_ugf
} // namespace sierra

#endif
