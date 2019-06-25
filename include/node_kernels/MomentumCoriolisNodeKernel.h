/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMCORIOLISSRCNODEKERNEL_H
#define MOMENTUMCORIOLISSRCNODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "CoriolisSrc.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;

class MomentumCoriolisNodeKernel : public NGPNodeKernel<MomentumCoriolisNodeKernel>
{
public:
  MomentumCoriolisNodeKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&);

  KOKKOS_FUNCTION
  MomentumCoriolisNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MomentumCoriolisNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  const CoriolisSrc cor_;

  ngp::Field<double> dualNodalVolume_;
  ngp::Field<double> densityNp1_;
  ngp::Field<double> velocityNp1_;

  unsigned dualNodalVolumeID_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned velocityNp1ID_ {stk::mesh::InvalidOrdinal};
};

}  // nalu
}  // sierra


#endif /* MOMENTUMCORIOLISSRCNODEKERNEL_H */
