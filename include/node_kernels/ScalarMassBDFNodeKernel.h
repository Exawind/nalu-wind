/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SCALARMASSBDFNODEKERNEL_H
#define SCALARMASSBDFNODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra{
namespace nalu{

class Realm;

class ScalarMassBDFNodeKernel : public NGPNodeKernel<ScalarMassBDFNodeKernel>
{
public:
  ScalarMassBDFNodeKernel(
    const stk::mesh::BulkData&,
    ScalarFieldType*);

  KOKKOS_FUNCTION
  ScalarMassBDFNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~ScalarMassBDFNodeKernel() = default;

  virtual void setup(Realm&) override;
  
  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  ngp::Field<double> scalarQNm1_;
  ngp::Field<double> scalarQN_;
  ngp::Field<double> scalarQNp1_;
  ngp::Field<double> densityNm1_;
  ngp::Field<double> densityN_;
  ngp::Field<double> densityNp1_;
  ngp::Field<double> dualNodalVolume_;

  unsigned scalarQNm1ID_ {stk::mesh::InvalidOrdinal};
  unsigned scalarQNID_ {stk::mesh::InvalidOrdinal};
  unsigned scalarQNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned densityNm1ID_ {stk::mesh::InvalidOrdinal};
  unsigned densityNID_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolumeID_ {stk::mesh::InvalidOrdinal};

  double dt_;
  double gamma1_, gamma2_, gamma3_;
};

} // namespace nalu
} // namespace Sierra

#endif
