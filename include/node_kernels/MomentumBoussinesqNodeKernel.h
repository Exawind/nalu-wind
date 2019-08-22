/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMBOUSSINESQNODEKERNEL_h
#define MOMENTUMBOUSSINESQNODEKERNEL_h

#include "node_kernels/NodeKernel.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra{
namespace nalu{

class SolutionOptions;

class MomentumBoussinesqNodeKernel : public NGPNodeKernel<MomentumBoussinesqNodeKernel>
{
public:
  MomentumBoussinesqNodeKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&);

  KOKKOS_FUNCTION
  MomentumBoussinesqNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MomentumBoussinesqNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  ngp::Field<double> dualNodalVolume_;
  ngp::Field<double> temperature_;
  const int nDim_;
  NodeKernelTraits::DblType tRef_;
  NodeKernelTraits::DblType rhoRef_;
  NodeKernelTraits::DblType beta_;

  unsigned dualNodalVolumeID_ {stk::mesh::InvalidOrdinal};
  unsigned temperatureID_ {stk::mesh::InvalidOrdinal};

  NALU_ALIGNED NodeKernelTraits::DblType gravity_[NodeKernelTraits::NDimMax];
};

} // namespace nalu
} // namespace Sierra

#endif
