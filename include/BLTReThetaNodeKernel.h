/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef BLTRETHETANODEKERNEL_H
#define BLTRETHETANODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {

class Realm;

class BLTReThetaNodeKernel : public NGPNodeKernel<BLTReThetaNodeKernel>
{
public:
  BLTReThetaNodeKernel(const stk::mesh::MetaData&);

  KOKKOS_FORCEINLINE_FUNCTION
  BLTReThetaNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~BLTReThetaNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;
 
  double Re_thetat(const double& Tu, const double& Fla);
  double F_lamda(const double& Tu, const double& lamda);
  double Secant_Re0tcor(const double& duds, const double& dens_local, const double& Tu, const double& visc);

private:
  ngp::Field<double> tke_;
  ngp::Field<double> sdr_;
  ngp::Field<double> density_;
  ngp::Field<double> visc_;
  ngp::Field<double> dudx_;
  ngp::Field<double> dkdx_;
  ngp::Field<double> dwdx_;
  ngp::Field<double> minD_;
  ngp::Field<double> dualNodalVolume_;
  ngp::Field<double> gamint_;
  ngp::Field<double> re0t_;
  ngp::Field<double> velocityNp1_;

  unsigned tkeID_             {stk::mesh::InvalidOrdinal};
  unsigned sdrID_             {stk::mesh::InvalidOrdinal};
  unsigned densityID_         {stk::mesh::InvalidOrdinal};
  unsigned viscID_            {stk::mesh::InvalidOrdinal};
  unsigned dudxID_            {stk::mesh::InvalidOrdinal};
  unsigned dkdxID_            {stk::mesh::InvalidOrdinal};
  unsigned dwdxID_            {stk::mesh::InvalidOrdinal};
  unsigned minDID_            {stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolumeID_ {stk::mesh::InvalidOrdinal};
  unsigned gamintID_          {stk::mesh::InvalidOrdinal};
  unsigned re0tID_            {stk::mesh::InvalidOrdinal};
  unsigned velocityNp1ID_     {stk::mesh::InvalidOrdinal};

  NodeKernelTraits::DblType c0t_;
  NodeKernelTraits::DblType ceTwo_;

  const int nDim_;
};

}  // nalu
}  // sierra


#endif
  
