// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MOMENTUMSSTAMSDIFFEDGEKERNEL_H
#define MOMENTUMSSTAMSDIFFEDGEKERNEL_H

#include "edge_kernels/EdgeKernel.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;

class MomentumSSTAMSDiffEdgeKernel
  : public NGPEdgeKernel<MomentumSSTAMSDiffEdgeKernel>
{
public:
  MomentumSSTAMSDiffEdgeKernel(
    const stk::mesh::BulkData&, const SolutionOptions&);

  MomentumSSTAMSDiffEdgeKernel() = delete;

  KOKKOS_FUNCTION
  virtual ~MomentumSSTAMSDiffEdgeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    EdgeKernelTraits::ShmemDataType&,
    const stk::mesh::FastMeshIndex&,
    const stk::mesh::FastMeshIndex&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> edgeAreaVec_;

  stk::mesh::NgpField<double> coordinates_;
  stk::mesh::NgpField<double> velocity_;
  stk::mesh::NgpField<double> tvisc_;
  stk::mesh::NgpField<double> density_;
  stk::mesh::NgpField<double> tke_;
  stk::mesh::NgpField<double> sdr_;
  stk::mesh::NgpField<double> beta_;
  stk::mesh::NgpField<double> nodalMij_;
  stk::mesh::NgpField<double> dudx_;
  stk::mesh::NgpField<double> avgVelocity_;
  stk::mesh::NgpField<double> avgDudx_;
  stk::mesh::NgpField<double> avgResAdeq_;

  unsigned edgeAreaVecID_{stk::mesh::InvalidOrdinal};
  unsigned coordinatesID_{stk::mesh::InvalidOrdinal};
  unsigned velocityID_{stk::mesh::InvalidOrdinal};
  unsigned turbViscID_{stk::mesh::InvalidOrdinal};
  unsigned densityNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned tkeNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned sdrNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned betaID_{stk::mesh::InvalidOrdinal};
  unsigned MijID_{stk::mesh::InvalidOrdinal};
  unsigned dudxID_{stk::mesh::InvalidOrdinal};
  unsigned avgVelocityID_{stk::mesh::InvalidOrdinal};
  unsigned avgDudxID_{stk::mesh::InvalidOrdinal};
  unsigned avgResAdeqID_{stk::mesh::InvalidOrdinal};

  const double includeDivU_;

  const double betaStar_;
  const double CMdeg_;
  const double aspectRatioSwitch_;

  const int nDim_;

  double relaxFacU_{1.0};
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMSSTAMSDIFFEDGEKERNEL_H */
