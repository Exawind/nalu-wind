/*------------------------------------------------------------------------*/
/*  Copyright 2019 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TURBKINETICENERGYRODINODEKERNEL_H          
#define TURBKINETICENERGYRODINODEKERNEL_H          

#include "node_kernels/NodeKernel.h"
#include "stk_ngp/Ngp.hpp"

namespace stk{
namespace mesh{
class MetaData;
}
}

namespace sierra{
namespace nalu{

class Realm;
class SolutionOptions;

class TurbKineticEnergyRodiNodeKernel : public NGPNodeKernel<TurbKineticEnergyRodiNodeKernel>
{
public:
  TurbKineticEnergyRodiNodeKernel(const stk::mesh::MetaData&, const SolutionOptions&);

  KOKKOS_FUNCTION
  TurbKineticEnergyRodiNodeKernel() = default;
  
  KOKKOS_FUNCTION
  virtual ~TurbKineticEnergyRodiNodeKernel() = default;

  virtual void setup(Realm &) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:

  ngp::Field<double> dhdx_;
  ngp::Field<double> specificHeat_;
  ngp::Field<double> tvisc_;
  ngp::Field<double> dualNodalVolume_;

  const unsigned dhdxID_           {stk::mesh::InvalidOrdinal};
  const unsigned specificHeatID_   {stk::mesh::InvalidOrdinal};
  const unsigned tviscID_          {stk::mesh::InvalidOrdinal};
  const unsigned dualNodalVolumeID_{stk::mesh::InvalidOrdinal};

  NALU_ALIGNED NodeKernelTraits::DblType gravity_[NodeKernelTraits::NDimMax];
  NodeKernelTraits::DblType       turbPr_;
  const NodeKernelTraits::DblType beta_;
  const int    nDim_;
};

} // namespace nalu
} // namespace Sierra

#endif
