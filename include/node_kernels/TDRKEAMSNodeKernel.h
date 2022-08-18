// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TDRKEAMSNODEKERNEL_H
#define TDRKEAMSNODEKERNEL_H

#include "node_kernels/NodeKernel.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;

class TDRKEAMSNodeKernel : public NGPNodeKernel<TDRKEAMSNodeKernel>
{
public:
  TDRKEAMSNodeKernel(const stk::mesh::MetaData&, const std::string);

  TDRKEAMSNodeKernel() = delete;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~TDRKEAMSNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> dualNodalVolume_;

  stk::mesh::NgpField<double> coordinates_;
  stk::mesh::NgpField<double> visc_;
  stk::mesh::NgpField<double> density_;
  stk::mesh::NgpField<double> tke_;
  stk::mesh::NgpField<double> tdr_;
  stk::mesh::NgpField<double> prod_;
  stk::mesh::NgpField<double> dplus_;
  stk::mesh::NgpField<double> wallDist_;

  unsigned dualNodalVolumeID_{stk::mesh::InvalidOrdinal};
  unsigned coordinatesID_{stk::mesh::InvalidOrdinal};
  unsigned viscID_{stk::mesh::InvalidOrdinal};
  unsigned tkeID_{stk::mesh::InvalidOrdinal};
  unsigned tdrID_{stk::mesh::InvalidOrdinal};
  unsigned dplusID_{stk::mesh::InvalidOrdinal};
  unsigned wallDistID_{stk::mesh::InvalidOrdinal};
  unsigned prodID_{stk::mesh::InvalidOrdinal};
  unsigned densityID_{stk::mesh::InvalidOrdinal};

  NodeKernelTraits::DblType cEpsOne_;
  NodeKernelTraits::DblType cEpsTwo_;
  NodeKernelTraits::DblType fOne_;
  const int nDim_;
};

} // namespace nalu
} // namespace sierra

#endif /* TDRKEAMSNODEKERNEL_H */
