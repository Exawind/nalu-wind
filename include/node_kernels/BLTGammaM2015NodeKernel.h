/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef BLTGAMMAM2015NODEKERNEL_H
#define BLTGAMMAM2015NODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {

class Realm;

class BLTGammaM2015NodeKernel : public NGPNodeKernel<BLTGammaM2015NodeKernel>
{
public:
  BLTGammaM2015NodeKernel(const stk::mesh::MetaData&);

  KOKKOS_FORCEINLINE_FUNCTION
  BLTGammaM2015NodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~BLTGammaM2015NodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

  double FPG(const double& out);

private:
  ngp::Field<double> tke_;
  ngp::Field<double> sdr_;
  ngp::Field<double> density_;
  ngp::Field<double> visc_;
  ngp::Field<double> dudx_;
  ngp::Field<double> minD_;
  ngp::Field<double> dualNodalVolume_;
  ngp::Field<double> coordinates_;
  ngp::Field<double> velocityNp1_;
  ngp::Field<double> gamint_;
  ngp::Field<double> gammaprod_;
  ngp::Field<double> gammasink_;
  ngp::Field<double> gammareth_;

  unsigned tkeID_             {stk::mesh::InvalidOrdinal};
  unsigned sdrID_             {stk::mesh::InvalidOrdinal};
  unsigned densityID_         {stk::mesh::InvalidOrdinal};
  unsigned viscID_           {stk::mesh::InvalidOrdinal};
  unsigned dudxID_            {stk::mesh::InvalidOrdinal};
  unsigned minDID_            {stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolumeID_ {stk::mesh::InvalidOrdinal};
  unsigned coordinatesID_     {stk::mesh::InvalidOrdinal};
  unsigned velocityNp1ID_     {stk::mesh::InvalidOrdinal};
  unsigned gamintID_       {stk::mesh::InvalidOrdinal};
  unsigned gammaprodID_       {stk::mesh::InvalidOrdinal};
  unsigned gammasinkID_       {stk::mesh::InvalidOrdinal};
  unsigned gammarethID_       {stk::mesh::InvalidOrdinal};

  NodeKernelTraits::DblType caOne_;
  NodeKernelTraits::DblType caTwo_;
  NodeKernelTraits::DblType ceOne_;
  NodeKernelTraits::DblType ceTwo_;

  int timeStepCount;

  const int nDim_;
};

}  // nalu
}  // sierra


#endif
  
