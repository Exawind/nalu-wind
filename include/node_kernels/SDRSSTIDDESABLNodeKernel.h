/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SDRSSTIDDESABLNODEKERNEL_H
#define SDRSSTIDDESABLNODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {

class Realm;

class SDRSSTIDDESABLNodeKernel : public NGPNodeKernel<SDRSSTIDDESABLNodeKernel>
{
public:
  SDRSSTIDDESABLNodeKernel(const stk::mesh::MetaData&);

  KOKKOS_FORCEINLINE_FUNCTION
  SDRSSTIDDESABLNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~SDRSSTIDDESABLNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  ngp::Field<double> tke_;
  ngp::Field<double> sdr_;
  ngp::Field<double> density_;
  ngp::Field<double> tvisc_;
  ngp::Field<double> dudx_;
  ngp::Field<double> dkdx_;
  ngp::Field<double> dwdx_;
  ngp::Field<double> wallDist_;
  ngp::Field<double> dualNodalVolume_;
  ngp::Field<double> fOneBlend_;
  ngp::Field<double> cellLengthScale_;


  unsigned tkeID_             {stk::mesh::InvalidOrdinal};
  unsigned sdrID_             {stk::mesh::InvalidOrdinal};
  unsigned densityID_         {stk::mesh::InvalidOrdinal};
  unsigned tviscID_           {stk::mesh::InvalidOrdinal};
  unsigned dudxID_            {stk::mesh::InvalidOrdinal};
  unsigned dkdxID_            {stk::mesh::InvalidOrdinal};
  unsigned dwdxID_            {stk::mesh::InvalidOrdinal};
  unsigned wallDistID_        {stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolumeID_ {stk::mesh::InvalidOrdinal};
  unsigned fOneBlendID_       {stk::mesh::InvalidOrdinal};
  unsigned cellLengthScaleID_ {stk::mesh::InvalidOrdinal};


  NodeKernelTraits::DblType betaStar_;
  NodeKernelTraits::DblType tkeProdLimitRatio_;
  NodeKernelTraits::DblType sigmaWTwo_;
  NodeKernelTraits::DblType betaOne_;
  NodeKernelTraits::DblType betaTwo_;
  NodeKernelTraits::DblType gammaOne_;
  NodeKernelTraits::DblType gammaTwo_;
  NodeKernelTraits::DblType relaxFac_;
  NodeKernelTraits::DblType cDESke_;
  NodeKernelTraits::DblType cDESkw_;
  NodeKernelTraits::DblType abl_bndtw_;
  NodeKernelTraits::DblType abl_deltandtw_;
    

  const int nDim_;
};

}  // nalu
}  // sierra


#endif /* SDRSSTNODEKERNEL_H */
