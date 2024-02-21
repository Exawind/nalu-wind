// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SDRSSTAMSNODEKERNEL_H
#define SDRSSTAMSNODEKERNEL_H

#include "node_kernels/NodeKernel.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;

class SDRSSTAMSNodeKernel : public NGPNodeKernel<SDRSSTAMSNodeKernel>
{
public:
  SDRSSTAMSNodeKernel(const stk::mesh::MetaData&, const std::string);

  SDRSSTAMSNodeKernel() = delete;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~SDRSSTAMSNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> dualNodalVolume_;

  stk::mesh::NgpField<double> coordinates_;
  stk::mesh::NgpField<double> viscosity_;
  stk::mesh::NgpField<double> tvisc_;
  stk::mesh::NgpField<double> rho_;
  stk::mesh::NgpField<double> tke_;
  stk::mesh::NgpField<double> sdr_;
  stk::mesh::NgpField<double> beta_;
  stk::mesh::NgpField<double> prod_;
  stk::mesh::NgpField<double> fOneBlend_;
  stk::mesh::NgpField<double> dkdx_;
  stk::mesh::NgpField<double> dwdx_;

  unsigned dualNodalVolumeID_{stk::mesh::InvalidOrdinal};
  unsigned coordinatesID_{stk::mesh::InvalidOrdinal};
  unsigned tviscID_{stk::mesh::InvalidOrdinal};
  unsigned tkeNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned sdrNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned betaID_{stk::mesh::InvalidOrdinal};
  unsigned fOneBlendID_{stk::mesh::InvalidOrdinal};
  unsigned dkdxID_{stk::mesh::InvalidOrdinal};
  unsigned dwdxID_{stk::mesh::InvalidOrdinal};
  unsigned prodID_{stk::mesh::InvalidOrdinal};
  unsigned densityID_{stk::mesh::InvalidOrdinal};

  NodeKernelTraits::DblType betaStar_;
  NodeKernelTraits::DblType sigmaWTwo_;
  NodeKernelTraits::DblType betaOne_;
  NodeKernelTraits::DblType betaTwo_;
  NodeKernelTraits::DblType tkeProdLimitRatio_;
  const int nDim_;

  bool lengthScaleLimiter_;
  NodeKernelTraits::DblType corfac_;
  NodeKernelTraits::DblType referenceVelocity_;
  NodeKernelTraits::DblType gammaOne_;
  NodeKernelTraits::DblType gammaTwo_;
  NodeKernelTraits::DblType sdrAmb_;
};

} // namespace nalu
} // namespace sierra

#endif /* SDRSSTAMSNODEKERNEL_H */
