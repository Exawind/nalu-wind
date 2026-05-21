// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#pragma once

#include "node_kernels/NodeKernel.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra::kynema_ugf {

class Realm;

class ScalarContResNodeKernel : public NGPNodeKernel<ScalarContResNodeKernel>
{
public:
  ScalarContResNodeKernel(const stk::mesh::BulkData&, ScalarFieldType*);

  virtual void setup(Realm&) final;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) final;

private:
  stk::mesh::NgpField<double> dnv_np1_;
  stk::mesh::NgpField<double> cont_res_;
  stk::mesh::NgpField<double> q_np1_;

  unsigned qID_{stk::mesh::InvalidOrdinal};
  unsigned dnvID_{stk::mesh::InvalidOrdinal};
  unsigned contresID_{stk::mesh::InvalidOrdinal};
};

} // namespace sierra::kynema_ugf
