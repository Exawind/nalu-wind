/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
/*
 * ActuatorDiskFAST.h
 *
 *  Created on: Oct 16, 2018
 *      Author: psakiev
 */

#ifndef ActuatorDiskFAST_h
#define ActuatorDiskFAST_h

#include "ActuatorFAST.h"
// TODO(psakiev):: Decide which is better, varying spreading, or varying
// sampling radially

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

/** Implementation of a periodic Bezier curve (Sanchez-Reyes, 2009) to connect
 * points at a specific radius The advantage of this method is it maps distorted
 * points to an elipsoide with fewer samples than pure B-Splines or Bezier
 * curves. Fewer points are needed to create a perfect circle in the case of
 * equispaced points (min =3) It is a parametric curve over the interval [0,
 * 2pi]
 */
class SweptPointLocator
{
public:
  SweptPointLocator();
  ~SweptPointLocator() = default;
  Point operator()(double t);
  void update_point_location(int i, Point p);
  static int binomial_coefficient(int n, int v);
  std::vector<Point> get_control_points();
  double get_radius(int pntNum);
  Point get_centriod();

private:
  const int order_ = 2; // fix order at 2 for 3 point sampling
  const double delta_ = 2.0 * std::acos(-1.0) / (order_ + 1);
  double periodic_basis(double t);
  void generate_control_points();
  std::vector<Point> bladePoints_;
  std::vector<Point> controlPoints_;
  bool controlPointsCurrent_;
};

} // namespace nalu
} // namespace sierra
#endif
