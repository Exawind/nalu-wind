// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

/*
 * ActuatorDiskFAST.h
 *
 *  Created on: Oct 16, 2018
 *      Author: psakiev
 */

#ifndef ActuatorDiskFAST_h
#define ActuatorDiskFAST_h

#include "ActuatorFAST.h"

namespace sierra {
namespace nalu {

/** Class to hold additional information for the disk points that
 *  that will need to be populated from the actuator line sampling
 *  in FAST
 *
 */
class ActuatorDiskFASTInfo : public ActuatorFASTInfo
{
public:
  int bladeSweptPts_{-1};
};
/** Class for an Actuator Disk
 *
 *  This class creates an actuator disk by sampling the flow field
 *  with an OpenFAST Actuator line model.  As a result it inherits the
 *  majority of its functionality from the ActuatorFAST class.
 *
 *  The procedure for the disk is:
 *
 *  1) After the actuator line data structures are created additional
 *     points are added in between the lines at each discrete radial location.
 *     The points are inserted with a periodic Bezier curve (Sanchez-Reyes,
 * 2009) which is a parametric curve over the interval t \in [0, 2pi]. Since
 * these points are in-between the actuator lines they are refered to as 'swept
 * points' throughout the code.
 *
 *     The Bezier function can capture the coning affects, and if desired (not
 * implemented now) distortion of the disk due to blade motion.  Currently the
 * point positions for the disk and lines are never updated.
 *
 *  2) During a run the actuator line proceeds as normal, but prior to forces
 * being sent back to nalu-wind the force is averaged between the blades at each
 * discrete radial location.  This average value is then spread evenly across
 * all the points at the given radius (swept and line points), and this is the
 * force that is sent back to nalu-wind at all the discrete points.
 *
 */
class ActuatorDiskFAST : public ActuatorFAST
{
public:
  ActuatorDiskFAST(Realm& realm, const YAML::Node& node);

  ~ActuatorDiskFAST() = default;

  void parse_disk_specific(const YAML::Node& node);

  void create_point_info_map_class_specific() override;

  void update_class_specific() override;

  void execute_class_specific(
    const int nDim,
    const stk::mesh::FieldBase* coordinates,
    stk::mesh::FieldBase* actuator_source,
    const stk::mesh::FieldBase* dual_nodal_volume) override;

  std::string get_class_name() override;

protected:
  Point get_blade_point_location(int turbineNum, int bladeNum, int radiusIndex);
  int
  number_of_swept_points(int numBlades, double radius, double targetArcLength);
  void add_swept_points_to_map();
  std::map<int, std::vector<std::vector<double>>> averageForcesMap_;
  std::map<std::size_t, int> pointRadiusMap_;
  std::map<int, std::vector<int>>
                               numSweptPointMap_; //{globTurbNo : numPoints between blades at each radius}
  bool useUniformAziSampling_{false};
};


} // namespace nalu
} // namespace sierra
#endif
