// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ScalarGclNodeKernel_h
#define ScalarGclNodeKernel_h

#include "node_kernels/NodeKernel.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

class VOFGclNodeKernel : public NGPNodeKernel<VOFGclNodeKernel>
{
public:
  VOFGclNodeKernel(const stk::mesh::BulkData&, ScalarFieldType*);

  KOKKOS_DEFAULTED_FUNCTION
  VOFGclNodeKernel() = default;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~VOFGclNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> scalarQNp1_;
  stk::mesh::NgpField<double> divV_;
  stk::mesh::NgpField<double> dualNdVolNm1_;
  stk::mesh::NgpField<double> dualNdVolN_;
  stk::mesh::NgpField<double> dualNdVolNp1_;

  unsigned scalarQNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned divVID_{stk::mesh::InvalidOrdinal};
  unsigned dualNdVolNm1ID_{stk::mesh::InvalidOrdinal};
  unsigned dualNdVolNID_{stk::mesh::InvalidOrdinal};
  unsigned dualNdVolNp1ID_{stk::mesh::InvalidOrdinal};

  double dt_;
  double gamma1_, gamma2_, gamma3_;
};

} // namespace nalu
} // namespace sierra

#endif
