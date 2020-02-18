/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef BLTGAMMANODEKERNEL_H
#define BLTGAMMANODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {

class Realm;

class BLTGammaNodeKernel : public NGPNodeKernel<BLTGammaNodeKernel>
{
public:
  BLTGammaNodeKernel(const stk::mesh::MetaData&);

  KOKKOS_FORCEINLINE_FUNCTION
  BLTGammaNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~BLTGammaNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

  double Comp_f_length(const double& Re0t);
  double Comp_Re_0c(const double& Re0t);

private:
  ngp::Field<double> tke_;
  ngp::Field<double> sdr_;
  ngp::Field<double> density_;
  ngp::Field<double> tvisc_;
  ngp::Field<double> visc_;
  ngp::Field<double> dudx_;
  ngp::Field<double> minD_;
  ngp::Field<double> dualNodalVolume_;
  ngp::Field<double> gamint_;
  ngp::Field<double> re0t_;

  unsigned tkeID_             {stk::mesh::InvalidOrdinal};
  unsigned sdrID_             {stk::mesh::InvalidOrdinal};
  unsigned densityID_         {stk::mesh::InvalidOrdinal};
  unsigned tviscID_           {stk::mesh::InvalidOrdinal};
  unsigned viscID_           {stk::mesh::InvalidOrdinal};
  unsigned dudxID_            {stk::mesh::InvalidOrdinal};
  unsigned minDID_            {stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolumeID_ {stk::mesh::InvalidOrdinal};
  unsigned gamintID_       {stk::mesh::InvalidOrdinal};
  unsigned re0tID_       {stk::mesh::InvalidOrdinal};

  NodeKernelTraits::DblType caOne_;
  NodeKernelTraits::DblType caTwo_;
  NodeKernelTraits::DblType ceOne_;
  NodeKernelTraits::DblType ceTwo_;

  const int nDim_;
};

}  // nalu
}  // sierra


#endif
  
