// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORSCALINGFLLC_H_
#define ACTUATORSCALINGFLLC_H_

#include <aero/actuator/ActuatorBulk.h>
#include <aero/actuator/ActuatorBulkSimple.h>
#ifdef NALU_USES_OPENFAST
#include <aero/actuator/ActuatorBulkFAST.h>
#endif
#include <stdexcept>

namespace sierra {
namespace nalu {
namespace FLLC {

template <typename range_type, typename helper_type>
void
scale_lift_force(
  ActuatorBulk& actBulk,
  const ActuatorMeta& actMeta,
  range_type& rangePolicy,
  helper_type& helper,
  const int offset,
  const int nPoints)
{
  // suppress compiler warnings for unused variables when compiling w/o openfast
  (void)offset;
  (void)nPoints;

  switch (actMeta.actuatorType_) {
  case (ActuatorType::ActLineSimpleNGP): {
    auto actBulkSimple = dynamic_cast<ActuatorBulkSimple&>(actBulk);
    auto actMetaSimple = dynamic_cast<const ActuatorMetaSimple&>(actMeta);
    auto G = helper.get_local_view(actBulkSimple.liftForceDistribution_);
    auto rho = helper.get_local_view(actBulkSimple.density_);

    const int turbId = actBulkSimple.localTurbineId_;
    double dR = actMetaSimple.dR_.h_view(turbId);

    Kokkos::parallel_for(
      "scale G", rangePolicy, ACTUATOR_LAMBDA(int i) {
        const double denom = rho(i) * dR;
        for (int j = 0; j < 3; ++j) {
          G(i, j) /= denom;
        }
      });
    break;
  }
  case (ActuatorType::ActLineFASTNGP):
  case (ActuatorType::ActDiskFASTNGP): {
#ifndef NALU_USES_OPENFAST
    ThrowErrorMsg("Actuator methods require OpenFAST");
#if !defined(KOKKOS_ENABLE_GPU)
    break;
#endif
#else
    auto G = helper.get_local_view(actBulk.liftForceDistribution_);
    auto point = helper.get_local_view(actBulk.pointCentroid_);
    Kokkos::parallel_for(
      "scale G FAST outputs", rangePolicy, ACTUATOR_LAMBDA(int i) {
        double dr = 0;
        if (i == offset) {
          for (int j = 0; j < 3; ++j) {
            dr += std::pow(point(i, j) - point(i + 1, j), 2.0);
          }
        } else if (i == offset + nPoints - 1) {
          for (int j = 0; j < 3; ++j) {
            dr += std::pow(point(i, j) - point(i - 1, j), 2.0);
          }
        } else {
          for (int j = 0; j < 3; ++j) {
            dr += std::pow(point(i - 1, j) - point(i + 1, j), 2.0);
          }
        }
        // dr is computed using central difference
        dr = 0.5 * std::sqrt(dr);
        for (int j = 0; j < 3; ++j) {
          G(i, j) /= dr;
        }
      });
    break;
#endif
  }
  default: {
    throw std::runtime_error("Unsupported actuator type supplied to the fllc");
  }
  }
}
} // namespace FLLC
} // namespace nalu
} // namespace sierra
#endif /* ACTUATORSCALINGFLLC_H_ */
