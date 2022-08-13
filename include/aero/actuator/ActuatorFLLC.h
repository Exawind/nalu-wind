// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORFLLC_H_
#define ACTUATORFLLC_H_

#include <aero/actuator/ActuatorTypes.h>
#include <aero/actuator/ActuatorBladeDistributor.h>
#include <vector>
#include <utility>

namespace sierra {
namespace nalu {

struct ActuatorBulk;
struct ActuatorMeta;

class FilteredLiftingLineCorrection
{
public:
  using exec_space = ActuatorFixedExecutionSpace;
  using mem_space = ActuatorFixedMemSpace;
  using mem_layout = ActuatorFixedMemLayout;

  FilteredLiftingLineCorrection(
    const ActuatorMeta& actMeta, ActuatorBulk& actBulk);
  FilteredLiftingLineCorrection() = delete;

  virtual ~FilteredLiftingLineCorrection(){};

  /**
   * @brief Compute the lift force distribution (G)
   * Compute equation 5.3 from Martinez-Tossas and Meneveau 2019
   * G^{n-1}(z_j) = 0.5 * c(z_j) U_\inf^2(z_j)C_L(z_j) [m^3/s^2]
   * since we don't get CL/CD from openfast, but rather computed force
   * we will extract lift from the total force by subtracting drag
   */
  void compute_lift_force_distribution();
  /**
   * @brief Compute gradient of the lift force distribution (\Delta G)
   * Compute equations 5.4 and 5.5 from Martinez-Tossas and Meneveau 2019
   */
  void grad_lift_force_distribution();

  /**
   * @brief Compute difference in induced velocities
   * Compute equation 5.7 from Martinez-Tossas and Meneveau 2019
   */
  void compute_induced_velocities();

  /**
   * @brief indicate if the fllc should be applied
   */
  bool is_active();

private:
  ActuatorBulk& actBulk_;
  const ActuatorMeta& actMeta_;
  std::vector<BladeDistributionInfo> bladeDistInfo_;
};
} // namespace nalu
} // namespace sierra

#endif /* ACTUATORFLLC_H_ */
