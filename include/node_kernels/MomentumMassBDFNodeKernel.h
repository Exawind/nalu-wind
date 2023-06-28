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
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

class MomentumMassBDFNodeKernel
  : public NGPNodeKernel<MomentumMassBDFNodeKernel>
{
public:
  MomentumMassBDFNodeKernel(const stk::mesh::BulkData&);

  KOKKOS_DEFAULTED_FUNCTION
  MomentumMassBDFNodeKernel() = default;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~MomentumMassBDFNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> velocityNm1_;
  stk::mesh::NgpField<double> velocityN_;
  stk::mesh::NgpField<double> velocityNp1_;
  stk::mesh::NgpField<double> densityNm1_;
  stk::mesh::NgpField<double> densityN_;
  stk::mesh::NgpField<double> densityNp1_;
  stk::mesh::NgpField<double> dpdx_;
  stk::mesh::NgpField<double> dnvNp1_;
  stk::mesh::NgpField<double> dnvN_;
  stk::mesh::NgpField<double> dnvNm1_;

  unsigned velocityNm1ID_{stk::mesh::InvalidOrdinal};
  unsigned velocityNID_{stk::mesh::InvalidOrdinal};
  unsigned velocityNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned densityNm1ID_{stk::mesh::InvalidOrdinal};
  unsigned densityNID_{stk::mesh::InvalidOrdinal};
  unsigned densityNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned dpdxID_{stk::mesh::InvalidOrdinal};
  unsigned dnvNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned dnvNID_{stk::mesh::InvalidOrdinal};
  unsigned dnvNm1ID_{stk::mesh::InvalidOrdinal};

  double dt_;
  int nDim_;
  double gamma1_, gamma2_, gamma3_;
  double solveIncompressibleEqn, om_solveIncompressibleEqn;
};

} // namespace nalu
} // namespace sierra

#endif
