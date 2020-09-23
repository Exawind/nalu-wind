/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TKESSTIDDESABLNODEKERNEL_H
#define TKESSTIDDESABLNODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"

namespace sierra {
namespace nalu {

class Realm;

class TKESSTIDDESABLNodeKernel : public NGPNodeKernel<TKESSTIDDESABLNodeKernel>
{
public:
  TKESSTIDDESABLNodeKernel(const stk::mesh::MetaData&);

  KOKKOS_FORCEINLINE_FUNCTION
  TKESSTIDDESABLNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~TKESSTIDDESABLNodeKernel() = default;

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
  stk::mesh::NgpField<double> dudx_;
  stk::mesh::NgpField<double> wallDist_;
  stk::mesh::NgpField<double> dualNodalVolume_;
  stk::mesh::NgpField<double> maxLenScale_;
  stk::mesh::NgpField<double> fOneBlend_;

  unsigned tkeID_             {stk::mesh::InvalidOrdinal};
  unsigned sdrID_             {stk::mesh::InvalidOrdinal};
  unsigned densityID_         {stk::mesh::InvalidOrdinal};
  unsigned viscID_            {stk::mesh::InvalidOrdinal};
  unsigned tviscID_           {stk::mesh::InvalidOrdinal};
  unsigned dudxID_            {stk::mesh::InvalidOrdinal};
  unsigned wallDistID_        {stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolumeID_ {stk::mesh::InvalidOrdinal};
  unsigned maxLenScaleID_     {stk::mesh::InvalidOrdinal};
  unsigned fOneBlendID_       {stk::mesh::InvalidOrdinal};

  NodeKernelTraits::DblType betaStar_;
  NodeKernelTraits::DblType tkeProdLimitRatio_;
  NodeKernelTraits::DblType cDESke_;
  NodeKernelTraits::DblType cDESkw_;
  NodeKernelTraits::DblType kappa_;    
  NodeKernelTraits::DblType iddes_Cw_;    
  NodeKernelTraits::DblType iddes_Cdt1_;    
  NodeKernelTraits::DblType iddes_Cdt2_;    
  NodeKernelTraits::DblType iddes_Cl_;    
  NodeKernelTraits::DblType iddes_Ct_;
  NodeKernelTraits::DblType abl_bndtw_;
  NodeKernelTraits::DblType abl_deltandtw_;
  NodeKernelTraits::DblType cEps_;
  NodeKernelTraits::DblType relaxFac_;

  const int nDim_;
};

}  // nalu
}  // sierra


#endif /* TKESSTIDDESABLNODEKERNEL_H */
