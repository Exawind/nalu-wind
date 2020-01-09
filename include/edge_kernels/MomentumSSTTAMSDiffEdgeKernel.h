// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef MOMENTUMSSTTAMSDIFFEDGEKERNEL_H
#define MOMENTUMSSTTAMSDIFFEDGEKERNEL_H

#include "edge_kernels/EdgeKernel.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;

class MomentumSSTTAMSDiffEdgeKernel
  : public NGPEdgeKernel<MomentumSSTTAMSDiffEdgeKernel>
{
public:
  MomentumSSTTAMSDiffEdgeKernel(
    const stk::mesh::BulkData&, const SolutionOptions&);

  KOKKOS_FUNCTION
  MomentumSSTTAMSDiffEdgeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MomentumSSTTAMSDiffEdgeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    EdgeKernelTraits::ShmemDataType&,
    const stk::mesh::FastMeshIndex&,
    const stk::mesh::FastMeshIndex&,
    const stk::mesh::FastMeshIndex&) override;

private:
  ngp::Field<double> edgeAreaVec_;

  ngp::Field<double> coordinates_;
  ngp::Field<double> velocity_;
  ngp::Field<double> tvisc_;
  ngp::Field<double> density_;
  ngp::Field<double> tke_;
  ngp::Field<double> sdr_;
  ngp::Field<double> alpha_;
  ngp::Field<double> nodalMij_;
  ngp::Field<double> dudx_;
  ngp::Field<double> avgVelocity_;
  ngp::Field<double> avgDudx_;

  unsigned edgeAreaVecID_{stk::mesh::InvalidOrdinal};
  unsigned coordinatesID_{stk::mesh::InvalidOrdinal};
  unsigned velocityID_{stk::mesh::InvalidOrdinal};
  unsigned turbViscID_{stk::mesh::InvalidOrdinal};
  unsigned densityNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned tkeNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned sdrNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned alphaID_{stk::mesh::InvalidOrdinal};
  unsigned MijID_{stk::mesh::InvalidOrdinal};
  unsigned dudxID_{stk::mesh::InvalidOrdinal};
  unsigned avgVelocityID_{stk::mesh::InvalidOrdinal};
  unsigned avgDudxID_{stk::mesh::InvalidOrdinal};

  const double includeDivU_;
  
  const double betaStar_;
  const double CMdeg_;

  const int nDim_;

  double relaxFacU_{1.0};
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMSSTTAMSDIFFEDGEKERNEL_H */
