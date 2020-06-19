// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MOMENTUMRayleighDampingNODEKERNEL_h
#define MOMENTUMRayleighDampingNODEKERNEL_h

#include "node_kernels/NodeKernel.h"

#include "SolutionOptions.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class MomentumRayleighDampingNodeKernel
  : public NGPNodeKernel<MomentumRayleighDampingNodeKernel>
{
public:
  MomentumRayleighDampingNodeKernel(
    const stk::mesh::MetaData&, RayleighDampingParameters, std::string);

  KOKKOS_FUNCTION
  MomentumRayleighDampingNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MomentumRayleighDampingNodeKernel() = default;

  virtual void setup(Realm&) override{};

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  using ftype = NodeKernelTraits::DblType;
  static constexpr int max_dim = 3;
  const int nDim_;

  stk::mesh::NgpField<double> volume_;
  stk::mesh::NgpField<double> velocity_;
  stk::mesh::NgpField<double> distance_;
  RayleighDampingParameters params_;
};

} // namespace nalu
} // namespace sierra

#endif
