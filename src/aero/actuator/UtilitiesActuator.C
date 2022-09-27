// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#include <cmath>

#include "aero/actuator/UtilitiesActuator.h" // master elements

// This is to access sierra::nalu::Coordinates
#include "NaluParsing.h"

namespace sierra {
namespace nalu {
namespace actuator_utils {

const double pi = M_PI;

#ifdef NALU_USES_OPENFAST
// The node ordering (from FAST) is
// Node 0 - Hub node
// Blade 1 nodes
// Blade 2 nodes
// Blade 3 nodes
// Tower nodes
Point
get_fast_point(
  fast::OpenFAST& fast,
  int turbId,
  fast::ActuatorNodeType type,
  int pointId,
  int bladeId)
{
  std::vector<double> coords(3);
  switch (type) {
  case fast::HUB: {
    fast.getForceNodeCoordinates(coords, 0, turbId);
    break;
  }
  case fast::TOWER: {
    const int offset =
      fast.get_numForcePts(turbId) - fast.get_numForcePtsTwr(turbId);
    fast.getForceNodeCoordinates(coords, pointId + offset, turbId);
    break;
  }
  case fast::BLADE: {
    const int nPBlade = fast.get_numForcePtsBlade(turbId);
    fast.getForceNodeCoordinates(
      coords, 1 + bladeId * nPBlade + pointId, turbId);
    break;
  }
  default: {
    break;
  }
  }
  return {coords[0], coords[1], coords[2]};
}

int
get_fast_point_index(
  const fast::fastInputs& fi,
  int turbId,
  int nBlades,
  fast::ActuatorNodeType type,
  int pointId,
  int bladeId)
{
  switch (type) {
  case fast::HUB: {
    return 0;
    break;
  }
  case fast::TOWER: {
    const int offset =
      fi.globTurbineData[turbId].numForcePtsBlade * nBlades + 1;
    return pointId + offset;
    break;
  }
  case fast::BLADE: {
    const int nPBlade = fi.globTurbineData[turbId].numForcePtsBlade;
    return 1 + bladeId * nPBlade + pointId;
    break;
  }
  default: {
    ThrowErrorMsg("Invalid fast type");
    return -1;
    break;
  }
  }
}
#endif

//--------------------------------------------------------------------------------------
//  Swept Point Locator
//--------------------------------------------------------------------------------------
SweptPointLocator::SweptPointLocator()
  : bladePoints_(3), controlPoints_(3), controlPointsCurrent_{false}
{
}

void
SweptPointLocator::update_point_location(int i, Point p)
{
  bladePoints_[i] = p;
  controlPointsCurrent_ = false;
}

// Set control points of the Bezier curve so that the blade points
// are on the resulting parametric curve.  This is ensured when the control
// points are 1) on the vector created by the blade point and centroid of blade
// points, 2) and are the blade points are the mid-points of lines connecting
// the control points. See figure 5 in the reference paper for clarification.
void
SweptPointLocator::generate_control_points()
{
  for (int d = 0; d < 3; d++) {
    controlPoints_[2][d] =
      bladePoints_[0][d] + bladePoints_[1][d] - bladePoints_[2][d];
    controlPoints_[1][d] = 2.0 * bladePoints_[1][d] - controlPoints_[2][d];
    controlPoints_[0][d] = 2.0 * bladePoints_[0][d] - controlPoints_[2][d];
  }
  controlPointsCurrent_ = true;
}

int
SweptPointLocator::binomial_coefficient(int N, int R)
{
  int coefficient{1};
  int upperLim = std::max(N - R, R);
  int lowerLim = std::min(N - R, R);
  for (int n = N; n > upperLim; n--) {
    coefficient *= n;
  }
  for (int r = lowerLim; r > 0; r--) {
    coefficient /= r;
  }
  return coefficient;
}

double
SweptPointLocator::periodic_basis(double t)
{
  int binom = binomial_coefficient(order_, order_ / 2);
  double denominator = static_cast<double>((order_ + 1) * binom);
  double eta = std::pow(2.0, order_) / denominator;
  return eta * std::pow(std::cos(0.5 * t), order_);
}

Point
SweptPointLocator::operator()(double t)
{
  Point output = {0, 0, 0};

  if (!controlPointsCurrent_) {
    generate_control_points();
  }

  for (int i = 0; i <= order_; i++) {
    const double offset = i * delta_;
    const double basis = periodic_basis(t - offset);
    for (int k = 0; k < 3; k++) {
      output[k] += controlPoints_[i][k] * basis;
    }
  }

  return output;
}

std::vector<Point>
SweptPointLocator::get_control_points()
{
  generate_control_points();
  return controlPoints_;
}

Point
SweptPointLocator::get_centriod()
{
  Point centroid = {0.0, 0.0, 0.0};

  for (int i = 0; i < 3; i++) {
    centroid[0] += bladePoints_[i][0];
    centroid[1] += bladePoints_[i][1];
    centroid[2] += bladePoints_[i][2];
  }

  centroid[0] /= 3.0;
  centroid[1] /= 3.0;
  centroid[2] /= 3.0;

  return centroid;
}

double
SweptPointLocator::get_radius(int pntNum)
{
  if (!controlPointsCurrent_) {
    generate_control_points();
  }
  double distance{0.0};
  Point centroid = get_centriod();

  for (int i = 0; i < 3; i++) {
    distance += std::pow(bladePoints_[pntNum][i] - centroid[i], 2.0);
  }

  return std::sqrt(distance);
}

///
/// A Gaussian projection function
///
double
Gaussian_projection(
  int nDim,                  // The dimension of the Gaussian (2 or 3)
  double* dis,               // The distance from the center of the Gaussian
  const Coordinates& epsilon // The width of the Gaussian
)
{
  // Compute the force projection weight at this location using a
  // Gaussian function.
  double g;
  if (nDim == 2)
    g = (1.0 / (epsilon.x_ * epsilon.y_ * pi)) *
        exp(-pow((dis[0] / epsilon.x_), 2.0) - pow((dis[1] / epsilon.y_), 2.0));
  else
    g = (1.0 / (epsilon.x_ * epsilon.y_ * epsilon.z_ * std::pow(pi, 1.5))) *
        exp(
          -pow((dis[0] / epsilon.x_), 2.0) - pow((dis[1] / epsilon.y_), 2.0) -
          pow((dis[2] / epsilon.z_), 2.0));

  return g;
}
///
/// A Gaussian projection function
///
double
Gaussian_projection(int nDim, double* dis, double* epsilon)
{
  // Compute the force projection weight at this location using a
  // Gaussian function.
  double g;
  if (nDim == 2)
    g = (1.0 / (epsilon[0] * epsilon[1] * pi)) *
        exp(-pow((dis[0] / epsilon[0]), 2.0) - pow((dis[1] / epsilon[1]), 2.0));
  else
    g = (1.0 / (epsilon[0] * epsilon[1] * epsilon[2] * std::pow(pi, 1.5))) *
        exp(
          -pow((dis[0] / epsilon[0]), 2.0) - pow((dis[1] / epsilon[1]), 2.0) -
          pow((dis[2] / epsilon[2]), 2.0));

  return g;
}

void
resize_std_vector(
  const int& sizeOfField,
  std::vector<double>& theVector,
  stk::mesh::Entity elem,
  const stk::mesh::BulkData& bulkData)
{
  const stk::topology& elemTopo = bulkData.bucket(elem).topology();
  MasterElement* meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element(elemTopo);
  const int nodesPerElement = meSCS->nodesPerElement_;
  theVector.resize(nodesPerElement * sizeOfField);
}

//--------------------------------------------------------------------------
//-------- gather_field ----------------------------------------------------
//--------------------------------------------------------------------------
void
gather_field(
  const int& sizeOfField,
  double* fieldToFill,
  const stk::mesh::FieldBase& stkField,
  stk::mesh::Entity const* elem_node_rels,
  const int& nodesPerElement)
{
  for (int ni = 0; ni < nodesPerElement; ++ni) {
    stk::mesh::Entity node = elem_node_rels[ni];
    const double* theField = (double*)stk::mesh::field_data(stkField, node);
    for (int j = 0; j < sizeOfField; ++j) {
      const int offSet = ni * sizeOfField + j;
      fieldToFill[offSet] = theField[j];
    }
  }
}

//--------------------------------------------------------------------------
//-------- gather_field_for_interp -----------------------------------------
//--------------------------------------------------------------------------
void
gather_field_for_interp(
  const int& sizeOfField,
  double* fieldToFill,
  const stk::mesh::FieldBase& stkField,
  stk::mesh::Entity const* elem_node_rels,
  const int& nodesPerElement)
{
  for (int ni = 0; ni < nodesPerElement; ++ni) {
    stk::mesh::Entity node = elem_node_rels[ni];
    const double* theField = (double*)stk::mesh::field_data(stkField, node);
    for (int j = 0; j < sizeOfField; ++j) {
      const int offSet = j * nodesPerElement + ni;
      fieldToFill[offSet] = theField[j];
    }
  }
}

//--------------------------------------------------------------------------
//-------- interpolate_field -----------------------------------------------
//--------------------------------------------------------------------------
void
interpolate_field(
  const int& sizeOfField,
  stk::mesh::Entity elem,
  const stk::mesh::BulkData& bulkData,
  const double* isoParCoords,
  const double* fieldAtNodes,
  double* pointField)
{
  // extract master element from the bucket in which the element resides
  const stk::topology& elemTopo = bulkData.bucket(elem).topology();
  MasterElement* meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element(elemTopo);

  // interpolate velocity to this best point
  meSCS->interpolatePoint(sizeOfField, isoParCoords, fieldAtNodes, pointField);
}

void
compute_distance(
  int nDim,
  const double* elemCentroid,
  const double* pointCentroid,
  double* distance)
{
  for (int j = 0; j < nDim; ++j) {
    distance[j] = elemCentroid[j] - pointCentroid[j];
  }
}

} // namespace actuator_utils
} // namespace nalu
} // namespace sierra
