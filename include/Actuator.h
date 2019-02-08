/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef Actuator_h
#define Actuator_h

#include <NaluParsing.h>
#include <FieldTypeDef.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Ghosting.hpp>
#include <stk_search/BoundingBox.hpp>
#include <stk_search/IdentProc.hpp>
#include <stk_search/SearchMethod.hpp>

// stk forwards
/*namespace stk {
  namespace mesh {
    struct Entity;
  }
  }*/

// basic c++
#include <string>
#include <vector>
#include <utility>

namespace sierra {
namespace nalu {

// common type defs
typedef stk::search::IdentProc<uint64_t, int> theKey;
typedef stk::search::Point<double> Point;
typedef stk::search::Sphere<double> Sphere;
typedef stk::search::Box<double> Box;
typedef std::pair<Sphere, theKey> boundingSphere;
typedef std::pair<Box, theKey> boundingElementBox;

class Realm;

class ActuatorInfo
{
public:
  ActuatorInfo() : processorId_(0), numPoints_(1), turbineName_("machine_one")
  {
  }
  virtual ~ActuatorInfo() {}

  int processorId_;
  int numPoints_;
  std::string turbineName_;
};

class ActuatorPointInfo
{
public:
  ActuatorPointInfo(
    Point centroidCoords,
    double searchRadius,
    double bestX,
    stk::mesh::Entity bestElem)
    : centroidCoords_(centroidCoords),
      searchRadius_(searchRadius),
      bestX_(bestX),
      bestElem_(bestElem)
  {
  }
  virtual ~ActuatorPointInfo() {}

  Point centroidCoords_; ///< The coordinates of the actuator point.
  double searchRadius_; ///< Elements within this search radius will be affected
                        ///< by this actuator point.
  double bestX_; ///< A number returned by stk::isInElement that determines
                 ///< whether an actuator point is inside (< 1) or outside an
                 ///< element (> 1). However, we choose the bestElem_ for this
                 ///< actuator point to be the one with the lowest bestX_.
  stk::mesh::Entity
    bestElem_; ///< The element within which the actuator point lies.
  std::vector<double>
    isoParCoords_; ///< The isoparametric coordinates of the bestElem_.
  std::set<stk::mesh::Entity>
    nodeVec_; ///< A list of nodes that are part of elements that lie within the
              ///< searchRadius_ around the actuator point.
};

class Actuator
{
public:
  Actuator(Realm& realm, const YAML::Node& node);

  virtual ~Actuator();

  // load all of the options
  virtual void load(const YAML::Node& node);

  // setup part creation and nodal field registration (before populate_mesh())
  virtual void setup() = 0;

  // setup part creation and nodal field registration (after populate_mesh())
  virtual void initialize() = 0;

  // populate nodal field and output norms (if appropriate)
  virtual void execute() = 0;

  // Use for polymorphic print statements
  virtual std::string get_class_name() = 0;

  //------------------------------------------------------------------

  // determine element bounding box in the mesh
  void populate_candidate_elements();

  // fill in the map that will hold point and ghosted elements
  void create_actuator_line_point_info_map();

  // figure out the set of elements that belong in the custom ghosting data
  // structure
  void determine_elems_to_ghost();

  // deal with custom ghosting
  void manage_ghosting();

  // populate vector of elements
  void complete_search();

  // support methods to gather data; scalar and vector
  void resize_std_vector(
    const int& sizeOfField,
    std::vector<double>& theVector,
    stk::mesh::Entity elem,
    const stk::mesh::BulkData& bulkData);

  // general gather methods for scalar and vector (both double)
  void gather_field(
    const int& sizeOfField,
    double* fieldToFill,
    const stk::mesh::FieldBase& stkField,
    stk::mesh::Entity const* elem_node_rels,
    const int& nodesPerElement);

  void gather_field_for_interp(
    const int& sizeOfField,
    double* fieldToFill,
    const stk::mesh::FieldBase& stkField,
    stk::mesh::Entity const* elem_node_rels,
    const int& nodesPerElement);

  // element volume and scv volume populated
  double compute_volume(
    const int& nDim,
    stk::mesh::Entity elem,
    const stk::mesh::BulkData& bulkData);

  // interpolate field to point centroid
  void interpolate_field(
    const int& sizeOfField,
    stk::mesh::Entity elem,
    const stk::mesh::BulkData& bulkData,
    const double* isoParCoords,
    const double* fieldAtNodes,
    double* pointField);

  double compute_distance(
    const int& nDim, const double* elemCentroid, const double* pointCentroid);

  //------------------------------------------------------------------
  // hold the realm
  Realm& realm_;

  // type of stk search
  stk::search::SearchMethod searchMethod_;

  // how many elements to ghost?
  uint64_t needToGhostCount_;
  stk::mesh::EntityProcVec elemsToGhost_;

  // save off product of search
  std::vector<std::pair<theKey, theKey>> searchKeyPair_;
  // bounding box data types for stk_search */
  std::vector<boundingSphere> boundingSphereVec_;
  std::vector<boundingElementBox> boundingElementBoxVec_;
  // target names for set of bounding boxes
  std::vector<std::string> searchTargetNames_;

  std::vector<std::unique_ptr<ActuatorInfo>> actuatorInfo_;
  std::map<size_t, std::unique_ptr<ActuatorPointInfo>> actuatorPointInfoMap_;

  // scratch space
  std::vector<double> ws_coordinates_;
  std::vector<double> ws_scv_volume_;
  std::vector<double> ws_velocity_;
  std::vector<double> ws_density_;
  std::vector<double> ws_viscosity_;

  stk::mesh::Ghosting* actuatorGhosting_;
};

} // namespace nalu
} // namespace sierra

#endif
