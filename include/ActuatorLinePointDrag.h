/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ActuatorLinePointDrag_h
#define ActuatorLinePointDrag_h

#include "Actuator.h"

namespace sierra {
namespace nalu {

class Realm;

class ActuatorLinePointDragInfo : public ActuatorInfo
{
public:
  ActuatorLinePointDragInfo();
  ~ActuatorLinePointDragInfo();

  // for each type of probe, e.g., line of site, hold some stuff
  double radius_;
  double omega_;
  double gaussDecayRadius_;
  Coordinates tipCoordinates_;
  Coordinates tailCoordinates_;
  Coordinates coordinates_;
};

// class that holds all of the action... for each point, hold the current
// location and other useful info
class ActuatorLinePointDragPointInfo : public ActuatorPointInfo
{
public:
  ActuatorLinePointDragPointInfo(
    Point centroidCoords,
    double radius,
    double omega,
    double twoSigSq,
    double* velocity);
  ~ActuatorLinePointDragPointInfo();
  double omega_;
  double gaussDecayRadius_;

  // mesh motion specifics
  double velocity_[3];
};

class ActuatorLinePointDrag : public Actuator
{
public:
  ActuatorLinePointDrag(Realm& realm, const YAML::Node& node);
  ~ActuatorLinePointDrag();

  // load all of the options
  void load(const YAML::Node& node) override;

  // setup part creation and nodal field registration (before populate_mesh())
  void setup() override;

  // setup part creation and nodal field registration (after populate_mesh())
  void initialize() override;

  // fill in the map that will hold point and ghosted elements
  void create_actuator_line_point_info_map();

  // manage rotation, now only in the y-z plane
  void set_current_coordinates(
    double* lineCentroid,
    double* centroidCoords,
    const double& omega,
    const double& currentTime);
  void set_current_velocity(
    double* lineCentroid,
    const double* centroidCoords,
    double* velocity,
    const double& omega);

  // populate nodal field and output norms (if appropriate)
  void execute() override;

  // drag at the point centroid
  void compute_point_drag(
    const int& nDim,
    const double& pointRadius,
    const double* pointVelocity,
    const double* pointGasVelocity,
    const double& pointGasViscosity,
    const double& pointGasDensity,
    double* pointDrag,
    double* pointDragLHS);

  // centroid of the element
  void compute_elem_centroid(
    const int& nDim, double* elemCentroid, const int& nodesPerElement);

  // drag fource at given radius
  void compute_node_drag_given_radius(
    const int& nDim,
    const double& radius,
    const double& epsilon,
    const double* pointDrag,
    double* nodeDrag);

  // finally, perform the assembly
  void assemble_lhs_to_best_elem_nodes(
    const int& nDim,
    stk::mesh::Entity elem,
    const stk::mesh::BulkData& bulkData,
    const double& elemVolume,
    const double* dragLHS,
    stk::mesh::FieldBase& actuator_source_lhs);

  // Spread the actuator force to a node vector
  void spread_actuator_force_to_node_vec(
    const int& nDim,
    const std::set<stk::mesh::Entity>& nodeVec,
    const std::vector<double>& actuator_force,
    const double* actuator_node_coordinates,
    const stk::mesh::FieldBase& coordinates,
    stk::mesh::FieldBase& actuator_source,
    const double& epsilon);

  // local id for set of points
  uint64_t localPointId_;

  // does the actuator line move?
  bool actuatorLineMotion_;

  // everyone needs pi
  const double pi_;

  std::string get_class_name() override;
};

} // namespace nalu
} // namespace sierra

#endif
