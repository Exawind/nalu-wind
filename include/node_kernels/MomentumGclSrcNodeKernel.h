/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MomentumGclSrcNodeKernel_h
#define MomentumGclSrcNodeKernel_h

#include "node_kernels/NodeKernel.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra{
namespace nalu{

class Realm;

class MomentumGclSrcNodeKernel : public NGPNodeKernel<MomentumGclSrcNodeKernel>
{
public:

  MomentumGclSrcNodeKernel(
    const stk::mesh::BulkData&);

  KOKKOS_FUNCTION
  MomentumGclSrcNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MomentumGclSrcNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  ngp::Field<double> velocityNp1_;
  ngp::Field<double> densityNp1_;
  ngp::Field<double> divV_;
  ngp::Field<double> dualNdVolNm1_;
  ngp::Field<double> dualNdVolN_;
  ngp::Field<double> dualNdVolNp1_;

  unsigned velocityNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned divVID_ {stk::mesh::InvalidOrdinal};
  unsigned dualNdVolNm1ID_ {stk::mesh::InvalidOrdinal};
  unsigned dualNdVolNID_ {stk::mesh::InvalidOrdinal};
  unsigned dualNdVolNp1ID_ {stk::mesh::InvalidOrdinal};

  int nDim_;
  double dt_;
  double gamma1_, gamma2_, gamma3_;
};

} // namespace nalu
} // namespace Sierra

#endif
