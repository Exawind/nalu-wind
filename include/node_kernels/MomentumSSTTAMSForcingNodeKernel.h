// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef MOMENTUMSSTTAMSFORCINGNODEKERNEL_H
#define MOMENTUMSSTTAMSFORCINGNODEKERNEL_H

#include "node_kernels/NodeKernel.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;

class MomentumSSTTAMSForcingNodeKernel
  : public NGPNodeKernel<MomentumSSTTAMSForcingNodeKernel>
{
public:
  MomentumSSTTAMSForcingNodeKernel(
    const stk::mesh::BulkData&, const SolutionOptions&);

  KOKKOS_FUNCTION
  MomentumSSTTAMSForcingNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MomentumSSTTAMSForcingNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  ngp::Field<double> dualNodalVolume_;

  ngp::Field<double> coordinates_;
  ngp::Field<double> velocity_;
  ngp::Field<double> viscosity_;
  ngp::Field<double> tvisc_;
  ngp::Field<double> density_;
  ngp::Field<double> tke_;
  ngp::Field<double> sdr_;
  ngp::Field<double> alpha_;
  ngp::Field<double> Mij_;
  ngp::Field<double> minDist_;
  ngp::Field<double> avgVelocity_;
  ngp::Field<double> avgTime_;
  ngp::Field<double> avgResAdeq_;

  unsigned dualNodalVolumeID_{stk::mesh::InvalidOrdinal};
  unsigned coordinatesID_{stk::mesh::InvalidOrdinal};
  unsigned velocityID_{stk::mesh::InvalidOrdinal};
  unsigned viscosityID_{stk::mesh::InvalidOrdinal};
  unsigned turbViscID_{stk::mesh::InvalidOrdinal};
  unsigned densityNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned tkeNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned sdrNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned alphaID_{stk::mesh::InvalidOrdinal};
  unsigned MijID_{stk::mesh::InvalidOrdinal};
  unsigned minDistID_{stk::mesh::InvalidOrdinal};
  unsigned avgVelocityID_{stk::mesh::InvalidOrdinal};
  unsigned avgResAdeqID_{stk::mesh::InvalidOrdinal};

  const double betaStar_;
  const double forceCl_;
  const double Ceta_;
  const double Ct_;
  const double blT_;
  const double blKol_;
  const double forceFactor_;
  const double cMu_; 
  const int nDim_;

  double pi_;
  double time_;
  double dt_;
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMSSTTAMSFORCINGNODEKERNEL_H */
