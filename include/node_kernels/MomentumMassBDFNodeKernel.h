// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef MomentumMassBDFNodeKernel_h
#define MomentumMassBDFNodeKernel_h

#include "node_kernels/NodeKernel.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra{
namespace nalu{

class Realm;

class MomentumMassBDFNodeKernel : public NGPNodeKernel<MomentumMassBDFNodeKernel>
{
public:
  MomentumMassBDFNodeKernel(
    const stk::mesh::BulkData&);

  KOKKOS_FUNCTION
  MomentumMassBDFNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MomentumMassBDFNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  ngp::Field<double> velocityNm1_;
  ngp::Field<double> velocityN_;
  ngp::Field<double> velocityNp1_;
  ngp::Field<double> densityNm1_;
  ngp::Field<double> densityN_;
  ngp::Field<double> densityNp1_;
  ngp::Field<double> dpdx_;
  ngp::Field<double> dualNodalVolume_;

  unsigned velocityNm1ID_ {stk::mesh::InvalidOrdinal};
  unsigned velocityNID_ {stk::mesh::InvalidOrdinal};
  unsigned velocityNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned densityNm1ID_ {stk::mesh::InvalidOrdinal};
  unsigned densityNID_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned dpdxID_ {stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolumeID_ {stk::mesh::InvalidOrdinal};
  
  double dt_;
  int nDim_;
  double gamma1_, gamma2_, gamma3_;
  
};

} // namespace nalu
} // namespace Sierra

#endif
