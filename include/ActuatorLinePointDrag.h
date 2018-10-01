/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef ActuatorLinePointDrag_h
#define ActuatorLinePointDrag_h

#include "Actuator.h"

namespace sierra{
namespace nalu{

class Realm;

class ActuatorLinePointDragInfo : public ActuatorInfo {
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

// class that holds all of the action... for each point, hold the current location and other useful info
class ActuatorLinePointDragPointInfo : public ActuatorPointInfo{
 public:
  ActuatorLinePointDragPointInfo(
    Point centroidCoords, double radius, double omega, double twoSigSq, double *velocity);
  ~ActuatorLinePointDragPointInfo();
  double omega_;
  double gaussDecayRadius_;

  // mesh motion specifics
  double velocity_[3];
};

 class ActuatorLinePointDrag: public Actuator
{
public:

  ActuatorLinePointDrag(
    Realm &realm,
    const YAML::Node &node);
  ~ActuatorLinePointDrag();

  // load all of the options
  void load(
    const YAML::Node & node);

  // setup part creation and nodal field registration (before populate_mesh())
  void setup();

  // setup part creation and nodal field registration (after populate_mesh())
  void initialize();

  // fill in the map that will hold point and ghosted elements
  void create_actuator_line_point_info_map();

  // figure out the set of elements that belong in the custom ghosting data structure
  void determine_elems_to_ghost();

  // deal with custom ghosting
  void manage_ghosting();

  // manage rotation, now only in the y-z plane
  void set_current_coordinates(
    double *lineCentroid, double *centroidCoords, const double &omega, const double &currentTime);
  void set_current_velocity(
    double *lineCentroid, const double *centroidCoords, double *velocity, const double &omega);

  // populate vector of elements
  void complete_search();

  // populate nodal field and output norms (if appropriate)
  void execute();

  // support methods to gather data; scalar and vector
  void resize_std_vector(
    const int &sizeOfField,
    std::vector<double> &theVector,
    stk::mesh::Entity elem,
    const stk::mesh::BulkData & bulkData);

  // general gather methods for scalar and vector (both double)
  void gather_field(
    const int &sizeOfField,
    double *fieldToFill,
    const stk::mesh::FieldBase &stkField,
    stk::mesh::Entity const* elem_node_rels,
    const int &nodesPerElement);

  void gather_field_for_interp(
    const int &sizeOfField,
    double *fieldToFill,
    const stk::mesh::FieldBase &stkField,
    stk::mesh::Entity const* elem_node_rels,
    const int &nodesPerElement);

  // element volume and scv volume populated
  double compute_volume(
    const int &nDim,
    stk::mesh::Entity elem,
    const stk::mesh::BulkData & bulkData);

  // interpolate field to point centroid
  void interpolate_field(
    const int &sizeOfField,
    stk::mesh::Entity elem,
    const stk::mesh::BulkData & bulkData,
    const double *isoParCoords,
    const double *fieldAtNodes,
    double *pointField);

  // drag at the point centroid
  void compute_point_drag(
    const int &nDim,
    const double &pointRadius,
    const double *pointVelocity,
    const double *pointGasVelocity,
    const double &pointGasViscosity,
    const double &pointGasDensity,
    double *pointDrag,
    double *pointDragLHS);

  // centroid of the element
  void compute_elem_centroid(
    const int &nDim,
    double *elemCentroid,
    const int &nodesPerElement);

  // radius from element centroid to point centroid
  double compute_radius(
    const int &nDim,
    const double *elemCentroid,
    const double *pointCentroid);

  // drag fource at given radius
  void compute_node_drag_given_radius(
    const int &nDim,
    const double &radius,
    const double &epsilon,
    const double *pointDrag,
    double *nodeDrag);

  // finally, perform the assembly
  void assemble_lhs_to_best_elem_nodes(
    const int &nDim,
    stk::mesh::Entity elem,
    const stk::mesh::BulkData & bulkData,
    const double &elemVolume,
    const double *dragLHS,
    stk::mesh::FieldBase &actuator_source_lhs);

  // Spread the actuator force to a node vector
  void spread_actuator_force_to_node_vec(
      const int &nDim,
      const std::set<stk::mesh::Entity>& nodeVec,
      const std::vector<double>& actuator_force,
      const double * actuator_node_coordinates,
      const stk::mesh::FieldBase & coordinates,
      stk::mesh::FieldBase & actuator_source,
      const double & epsilon);


  // custom ghosting
  stk::mesh::Ghosting *actuatorLineGhosting_;


  // local id for set of points
  uint64_t localPointId_;

  // does the actuator line move?
  bool actuatorLineMotion_;

  // everyone needs pi
  const double pi_;

//DELETEME
  // vector of averaging information
   std::vector<ActuatorLinePointDragInfo *> actuatorLineInfo_;

  // map of point info objects
   std::map<size_t, ActuatorLinePointDragPointInfo *> actuatorLinePointInfoMap_;

  std::string get_class_name() override;
};


} // namespace nalu
} // namespace Sierra

#endif
