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

#include <actuator/ActuatorTypes.h>

namespace sierra {
namespace nalu {

struct ActuatorBulk;
struct ActuatorMeta;

namespace FLLC {
/**
 * @brief Compute the lift force distribution (G)
 * Compute equation 5.3 from Martinez-Tossas and Meneveau 2019
 * G^{n-1}(z_j) = 0.5 * c(z_j) U_\inf^2(z_j)C_L(z_j) [m^3/s^2]
 * since we don't get CL/CD from openfast, but rather computed force
 * we will extract lift from the total force by subtracting drag
 *
 * @param actBulk - Container to hold all the fields
 * @param actMeta - Container for general turbine info
 */
void compute_lift_force_distribution(
  ActuatorBulk& actBulk, const ActuatorMeta& actMeta);
/**
 * @brief Compute gradient of the lift force distribution (\Delta G)
 * Compute equations 5.4 and 5.5 from Martinez-Tossas and Meneveau 2019
 *
 * @param actBulk
 * @param actMeta
 */
void grad_lift_force_distribution(
  ActuatorBulk& actBulk, const ActuatorMeta& actMeta);

/**
 * @brief Compute difference in induced velocities
 * Compute equation 5.7 from Martinez-Tossas and Meneveau 2019
 *
 * @param actBulk
 * @param actMeta
 */
// void
// compute_induced_velocities(ActuatorBulk& actBulk, const ActuatorMeta&
// actMeta);

void
compute_induced_velocities(ActuatorBulk& actBulk, const ActuatorMeta& actMeta);

} // namespace FLLC

void Compute_FLLC(ActuatorBulk& actBulk, const ActuatorMeta& actMeta);
void Apply_FLLC(ActuatorBulk& actBulk, const ActuatorMeta& actMeta);

} // namespace nalu
} // namespace sierra

#endif /* ACTUATORFLLC_H_ */
