// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MOMENTUMBODYFORCEBOXNODEKERNEL_H
#define MOMENTUMBODYFORCEBOXNODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "SolutionOptions.h"
#include "EquationSystems.h"
#include "EquationSystem.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

#include "ngp_algorithms/GeometryAlgDriver.h"
#include "ngp_algorithms/GeometryBoundaryAlg.h"
#include "ngp_algorithms/MdotAlgDriver.h"
#include "ngp_algorithms/MdotInflowAlg.h"

namespace sierra {
namespace nalu {

class MomentumBodyForceBoxNodeKernel
  : public NGPNodeKernel<MomentumBodyForceBoxNodeKernel>
{
public:
  MomentumBodyForceBoxNodeKernel(
    Realm& realm,
    const std::vector<double>&,
    const std::vector<double>& = std::vector<double>());

  KOKKOS_FUNCTION
  MomentumBodyForceBoxNodeKernel() = delete;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~MomentumBodyForceBoxNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> coordinates_;
  stk::mesh::NgpField<double> dualNodalVolume_;

  NodeKernelTraits::DblType forceVector_[NodeKernelTraits::NDimMax];
  NodeKernelTraits::DblType lo_[NodeKernelTraits::NDimMax];
  NodeKernelTraits::DblType hi_[NodeKernelTraits::NDimMax];
  stk::mesh::Part* mdotPart_;
  GeometryAlgDriver* geometryAlgDriver_;
  MdotAlgDriver* mdotAlgDriver_;

  const int nDim_;
  unsigned coordinatesID_{stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolumeID_{stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVecID_{stk::mesh::InvalidOrdinal};
  unsigned pressureForceID_{stk::mesh::InvalidOrdinal};
  unsigned viscousForceID_{stk::mesh::InvalidOrdinal};
  const std::string& outputFileName_;
  const bool& dynamic_;
  const int w_;
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMBODYFORCEBOXNODEKERNEL_H */
