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
#include "gtest/gtest_prod.h"
//TODO(psakiev):: Function to define radial sample points line (cone)
//TODO(psakiev):: Decide which is better, varying spreading, or varying sampling radially

/*
 * Implementation:
 * -------------------------------------------------------------
 * Put disk points into normal point map after the normal points
 * for actuator lines are computed.
 * The points are mapped between the actuator line points
 * -------------------------------------------------------------
 *    - pros:
 *        * execution of force spreading is the same
 *        * ghosting, search same
 *    - cons:
 *        * have to figure out a way to only send a portion of the points to fast
 */

namespace sierra{
namespace nalu{

/** Class to hold additional information for the disk points that
 *  that will need to be populated from the actuator line sampling
 *  in FAST
 *
 */
class ActuatorDiskFASTInfo : public ActuatorFASTInfo{
public:
  ActuatorDiskFASTInfo();
  ~ActuatorDiskFASTInfo()=default;
  int bladeSweptPts_;
};

class ActuatorDiskFAST : public ActuatorFAST{
public:
  ActuatorDiskFAST( Realm &realm, const YAML::Node &node);

  ~ActuatorDiskFAST()=default;

  void parse_disk_specific( const YAML::Node& node);

  void create_point_info_map_class_specific() override;

  void update_class_specific() override;

  void execute_class_specific(
    const int nDim,
    const stk::mesh::FieldBase * coordinates,
    stk::mesh::FieldBase * actuator_source,
    const stk::mesh::FieldBase * dual_nodal_volume
    ) override;

  std::string get_class_name() override;

protected:
  std::map<int,std::vector<std::vector<double>>> loadAverageMap_;
  std::map<std::size_t,int> pointRadiusMap_;
  std::map<int,int> sweptPointMap_; //{globTurbNo : num points between actuator lines}
};

/** Implementation of a periodic Bezier curve (Sanchez-Reyes, 2009) to connect points at a specific radius
 *  The advantage of this method is it maps distorted points to an elipsoide with fewer samples than pure B-Splines
 *  or Bezier curves. Fewer points are needed to create a perfect circle in the case of equispaced points (min =3)
 *  It is a parametric curve over the interval [0, 2pi]
 */
class SweptPointLocator{
public:
  SweptPointLocator();
  ~SweptPointLocator()=default;
  Point operator()(double t);
  void update_point_location(int i, Point p);
  static int binomial_coefficient(int n, int v);
  std::vector<Point> get_control_points();
private:
  const int order_ = 2; // fix order at 2 for 3 point sampling
  const double delta_ = 2.0*std::acos(-1.0)/(order_+1);
  double periodic_basis(double t);
  void generate_control_points();
  std::vector<Point> bladePoints_;
  std::vector<Point> controlPoints_;
  bool controlPointsCurrent_;

  FRIEND_TEST(ActuatorDiskFAST,SweptPointLocatorBasis);

};

// free functions
int ComputeNumberOfSweptPoints(double radius, int numBlades, double targetArcLength);
int FindClosestIndex(const double radius, std::vector<double> vecRad);
std::vector<double> NormalizedDirection(const std::vector<double>& p1, const std::vector<double>& p2);
Point SweptPointLocation(
  const int nBlades,
  const int nTotalPoints,
  const int nCurrentPoint,
  const Point& pointLoc,
  const std::vector<double>& hubLoc,
  const std::vector<double>& axis);

} // namespace nalu
} // namespace sierra
#endif
