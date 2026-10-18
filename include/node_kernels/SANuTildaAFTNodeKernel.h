// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#ifndef SANUTILDAAFTNODEKERNEL_H
#define SANUTILDAAFTNODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "FieldTypeDef.h"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace kynema_ugf {

class Realm;

class SANuTildaAFTNodeKernel : public NGPNodeKernel<SANuTildaAFTNodeKernel>
{
public:
  SANuTildaAFTNodeKernel(const stk::mesh::MetaData&);

  SANuTildaAFTNodeKernel() = delete;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~SANuTildaAFTNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> nuTilda_;
  stk::mesh::NgpField<double> aftGammaTilda_;
  stk::mesh::NgpField<double> density_;
  stk::mesh::NgpField<double> viscosity_;
  stk::mesh::NgpField<double> dudx_;
  stk::mesh::NgpField<double> dnutdx_;
  stk::mesh::NgpField<double> dualNodalVolume_;
  stk::mesh::NgpField<double> minDistance_;

  unsigned nuTildaID_{stk::mesh::InvalidOrdinal};
  unsigned aftGammaTildaID_{stk::mesh::InvalidOrdinal};
  unsigned densityID_{stk::mesh::InvalidOrdinal};
  unsigned viscosityID_{stk::mesh::InvalidOrdinal};
  unsigned dudxID_{stk::mesh::InvalidOrdinal};
  unsigned dnutdxID_{stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolumeID_{stk::mesh::InvalidOrdinal};
  unsigned minDistanceID_{stk::mesh::InvalidOrdinal};

  NodeKernelTraits::DblType Cb1_;
  NodeKernelTraits::DblType Cb2_;
  NodeKernelTraits::DblType sigma_;
  NodeKernelTraits::DblType kappa_;
  NodeKernelTraits::DblType Cv1_;
  NodeKernelTraits::DblType Cv2_;
  NodeKernelTraits::DblType Cv3_;
  NodeKernelTraits::DblType Cw2_;
  NodeKernelTraits::DblType Cw3_;
  NodeKernelTraits::DblType Ct3_;
  NodeKernelTraits::DblType Cw1_;

  const int nDim_;
};

} // namespace kynema_ugf
} // namespace sierra

#endif /* SANUTILDAAFTNODEKERNEL_H */