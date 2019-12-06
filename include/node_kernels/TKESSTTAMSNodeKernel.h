// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef TKESSTTAMSNODEKERNEL_H
#define TKESSTTAMSNODEKERNEL_H

#include "node_kernels/NodeKernel.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;

class TKESSTTAMSNodeKernel : public NGPNodeKernel<TKESSTTAMSNodeKernel>
{
public:
  TKESSTTAMSNodeKernel(const stk::mesh::MetaData&, const std::string);

  KOKKOS_FUNCTION
  TKESSTTAMSNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~TKESSTTAMSNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  ngp::Field<double> dualNodalVolume_;

  ngp::Field<double> coordinates_;
  ngp::Field<double> viscosity_;
  ngp::Field<double> tvisc_;
  ngp::Field<double> rho_;
  ngp::Field<double> tke_;
  ngp::Field<double> sdr_;
  ngp::Field<double> prod_;

  unsigned dualNodalVolumeID_{stk::mesh::InvalidOrdinal};
  unsigned coordinatesID_{stk::mesh::InvalidOrdinal};
  unsigned viscID_{stk::mesh::InvalidOrdinal};
  unsigned tviscID_{stk::mesh::InvalidOrdinal};
  unsigned tkeNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned sdrNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned prodID_{stk::mesh::InvalidOrdinal};
  unsigned densityID_{stk::mesh::InvalidOrdinal};

  NodeKernelTraits::DblType betaStar_;
  NodeKernelTraits::DblType tkeProdLimitRatio_;
  const int nDim_;
};

} // namespace nalu
} // namespace sierra

#endif /* TKESSTTAMSNODEKERNEL_H */
