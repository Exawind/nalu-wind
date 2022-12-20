// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef VOFMASSBDFNODEKERNEL_H
#define VOFMASSBDFNODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

class VOFMassBDFNodeKernel : public NGPNodeKernel<VOFMassBDFNodeKernel>
{
public:
  VOFMassBDFNodeKernel(const stk::mesh::BulkData&, ScalarFieldType*);

  KOKKOS_DEFAULTED_FUNCTION
  VOFMassBDFNodeKernel() = default;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~VOFMassBDFNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> scalarQNm1_;
  stk::mesh::NgpField<double> scalarQN_;
  stk::mesh::NgpField<double> scalarQNp1_;
  stk::mesh::NgpField<double> dnvNp1_;
  stk::mesh::NgpField<double> dnvN_;
  stk::mesh::NgpField<double> dnvNm1_;

  unsigned scalarQNm1ID_{stk::mesh::InvalidOrdinal};
  unsigned scalarQNID_{stk::mesh::InvalidOrdinal};
  unsigned scalarQNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned dnvNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned dnvNID_{stk::mesh::InvalidOrdinal};
  unsigned dnvNm1ID_{stk::mesh::InvalidOrdinal};

  double dt_;
  double gamma1_, gamma2_, gamma3_;
};

} // namespace nalu
} // namespace sierra

#endif
