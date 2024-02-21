// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MOMENTUMKEAMSFORCINGNODEKERNEL_H
#define MOMENTUMKEAMSFORCINGNODEKERNEL_H

#include "node_kernels/NodeKernel.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;

class MomentumKEAMSForcingNodeKernel
  : public NGPNodeKernel<MomentumKEAMSForcingNodeKernel>
{
public:
  MomentumKEAMSForcingNodeKernel(
    const stk::mesh::BulkData&, const SolutionOptions&);

  MomentumKEAMSForcingNodeKernel() = delete;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~MomentumKEAMSForcingNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> dualNodalVolume_;

  stk::mesh::NgpField<double> coordinates_;
  stk::mesh::NgpField<double> velocity_;
  stk::mesh::NgpField<double> viscosity_;
  stk::mesh::NgpField<double> tvisc_;
  stk::mesh::NgpField<double> density_;
  stk::mesh::NgpField<double> tke_;
  stk::mesh::NgpField<double> tdr_;
  stk::mesh::NgpField<double> beta_;
  stk::mesh::NgpField<double> Mij_;
  stk::mesh::NgpField<double> minDist_;
  stk::mesh::NgpField<double> avgVelocity_;
  stk::mesh::NgpField<double> avgTime_;
  stk::mesh::NgpField<double> avgResAdeq_;
  stk::mesh::NgpField<double> forcingComp_;

  unsigned dualNodalVolumeID_{stk::mesh::InvalidOrdinal};
  unsigned coordinatesID_{stk::mesh::InvalidOrdinal};
  unsigned velocityID_{stk::mesh::InvalidOrdinal};
  unsigned viscosityID_{stk::mesh::InvalidOrdinal};
  unsigned turbViscID_{stk::mesh::InvalidOrdinal};
  unsigned densityNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned tkeNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned tdrNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned betaID_{stk::mesh::InvalidOrdinal};
  unsigned MijID_{stk::mesh::InvalidOrdinal};
  unsigned minDistID_{stk::mesh::InvalidOrdinal};
  unsigned avgVelocityID_{stk::mesh::InvalidOrdinal};
  unsigned avgResAdeqID_{stk::mesh::InvalidOrdinal};
  unsigned forcingCompID_{stk::mesh::InvalidOrdinal};

  const double betaStar_;
  const double forceCl_;
  const double Ceta_;
  const double Ct_;
  const double blT_;
  const double blKol_;
  const double forceFactor_;
  const double cMu_;
  const double periodicForcingLengthX_;
  const double periodicForcingLengthY_;
  const double periodicForcingLengthZ_;
  const int nDim_;

  double time_;
  double dt_;
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMKEAMSFORCINGNODEKERNEL_H */
