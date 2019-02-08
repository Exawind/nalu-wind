/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <ActuatorLinePointDrag.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>
#include <NaluEnv.h>
#include <Realm.h>
#include <Simulation.h>
#include <nalu_make_unique.h>

// master elements
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

// stk_search
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>

// basic c++
#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <cmath>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// ActuatorLinePointDragInfo - holds all points in the tower specification
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ActuatorLinePointDragInfo::ActuatorLinePointDragInfo()
  : ActuatorInfo(), radius_(0), omega_(0.0), gaussDecayRadius_(1.5)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ActuatorLinePointDragInfo::~ActuatorLinePointDragInfo()
{
  // nothing to do
}

//==========================================================================
// Class Definition
//==========================================================================
// ActuatorLinePointDragPointInfo - holds individual points information
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ActuatorLinePointDragPointInfo::ActuatorLinePointDragPointInfo(
  Point centroidCoords,
  double radius,
  double omega,
  double gaussDecayRadius,
  double* velocity)
  : ActuatorPointInfo(centroidCoords, radius, 1.0e16, stk::mesh::Entity()),
    omega_(omega),
    gaussDecayRadius_(gaussDecayRadius)
{
  // initialize point velocity and displacement
  velocity_[0] = velocity[0];
  velocity_[1] = velocity[1];
  velocity_[2] = velocity[2];
}
//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ActuatorLinePointDragPointInfo::~ActuatorLinePointDragPointInfo()
{
  // nothing to do
}

//==========================================================================
// Class Definition
//==========================================================================
// ActuatorLinePointDrag - assemble source term for subgrin turbine; WIP
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ActuatorLinePointDrag::ActuatorLinePointDrag(
  Realm& realm, const YAML::Node& node)
  : Actuator(realm, node),
    localPointId_(0),
    actuatorLineMotion_(false),
    pi_(acos(-1.0))
{
  // load the data
  load(node);

  /*
    current WIP prototype
    Design concepts:
     1) First and foremost, elements are ghosted to the owning point rank. This
        probably should be changed since the number of elements might be larger
        than the number of points. Therefore, ghosting points to elements is
    probably easier. This will remove the parallel sum contributions from
    ghosted elements. time will tell..

     2) There can be many specifications with the number of points and omega
    processed.

     3) in the end, we fill the map of ActuatorLinePointDragPointInfo objects
    and iterate this guy to assemble source terms

     4) at present, fake source terms on simple Gaussian weighting

    actuator:
      search_method: stk_octree
      search_target_part: block_1

      specifications:

        - name: machine_one
          radius: 2.0
          omega: 1.0
          gaussian_decay_radius: 1.5
          tip_coodinates: [0.0, 0.0, 0.0]
          tail_coodinates: [0.0, 0.0, 0.0]
  */
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ActuatorLinePointDrag::~ActuatorLinePointDrag() {}

//--------------------------------------------------------------------------
//-------- compute_point_drag ----------------------------------------------
//--------------------------------------------------------------------------
void
ActuatorLinePointDrag::compute_point_drag(
  const int& nDim,
  const double& pointRadius,
  const double* pointVelocity,
  const double* pointGasVelocity,
  const double& pointGasViscosity,
  const double& pointGasDensity,
  double* pointForce,
  double* pointForceLHS)
{
  // compute magnitude of velocity difference between point and gas
  double vRelMag = 0.0;
  for (int j = 0; j < nDim; ++j)
    vRelMag += (pointVelocity[j] - pointGasVelocity[j]) *
               (pointVelocity[j] - pointGasVelocity[j]);
  vRelMag = std::sqrt(vRelMag);

  // Reynolds number and friction factors
  double ReP =
    2.0 * pointGasDensity * pointRadius * vRelMag / pointGasViscosity;
  double CubeRtReP = (ReP < 1000.) ? std::cbrt(ReP) : 0.0;
  double fD =
    (ReP < 1000.0) ? (1.0 + CubeRtReP * CubeRtReP / 6.0) : (0.424 / 24.0 * ReP);
  double coef = 6.0 * pi_ * pointGasViscosity * pointRadius;

  // this is from the fluids perspective, not the psuedo particle

  for (int j = 0; j < nDim; ++j) {
    pointForce[j] = coef * fD * (pointVelocity[j] - pointGasVelocity[j]);
    pointForceLHS[j] = coef * fD;
  }
}

//--------------------------------------------------------------------------
//-------- compute_node_drag_given_radius ----------------------------------
//--------------------------------------------------------------------------
void
ActuatorLinePointDrag::compute_node_drag_given_radius(
  const int& nDim,
  const double& radius,
  const double& epsilon,
  const double* pointForce,
  double* nodeDrag)
{
  // gaussian weight based on radius
  double gaussWeight;
  const double pi = acos(-1.0);
  if (nDim == 2)
    gaussWeight = (1.0 / (epsilon * epsilon * pi)) *
                  exp(-radius * radius / epsilon * epsilon);
  else
    gaussWeight = (1.0 / (epsilon * epsilon * epsilon * pow(pi, 1.5))) *
                  exp(-radius * radius / epsilon * epsilon);

  for (int j = 0; j < nDim; ++j)
    nodeDrag[j] = pointForce[j] * gaussWeight;
}

//--------------------------------------------------------------------------
//-------- load ------------------------------------------------------------
//--------------------------------------------------------------------------
void
ActuatorLinePointDrag::load(const YAML::Node& y_node)
{
  Actuator::load(y_node);
  // check for any data probes
  const YAML::Node y_actuatorLine = y_node["actuator"];
  if (y_actuatorLine) {
    const YAML::Node y_specs =
      expect_sequence(y_actuatorLine, "specifications", false);
    if (y_specs) {

      // save off number of towers
      const int numTowers = y_specs.size();

      // deal with processors... Distribute each tower over subsequent procs
      const int numProcs = NaluEnv::self().parallel_size();
      const int divProcTower = std::max(numProcs / numTowers, numProcs);

      // each specification can have multiple machines
      for (size_t ispec = 0; ispec < y_specs.size(); ++ispec) {
        const YAML::Node y_spec = y_specs[ispec];

        actuatorInfo_.emplace_back(new ActuatorLinePointDragInfo());
        auto actuatorLineInfo =
          dynamic_cast<ActuatorLinePointDragInfo*>(actuatorInfo_.back().get());

        // name
        const YAML::Node theName = y_spec["turbine_name"];
        if (theName)
          actuatorLineInfo->turbineName_ = theName.as<std::string>();
        else
          throw std::runtime_error("ActuatorLinePointDrag: no name provided");

        // processor id; distribute los equally over the number of processors
        actuatorLineInfo->processorId_ =
          divProcTower > 0 ? ispec % divProcTower : 0;

        // number of points
        get_if_present(
          y_spec, "number_of_points", actuatorLineInfo->numPoints_,
          actuatorLineInfo->numPoints_);
        if (actuatorLineInfo->numPoints_ < 2)
          throw std::runtime_error("ActuatorLinePointDrag: number of points "
                                   "must have at least two points");

        // radius and omega
        get_if_present(
          y_spec, "radius", actuatorLineInfo->radius_,
          actuatorLineInfo->radius_);
        get_if_present(
          y_spec, "omega", actuatorLineInfo->omega_, actuatorLineInfo->omega_);
        if (actuatorLineInfo->omega_ > 0.0 || actuatorLineInfo->omega_ < 0.0)
          actuatorLineMotion_ = true;

        // Gaussian props
        get_if_present(
          y_spec, "gaussian_decay_radius", actuatorLineInfo->gaussDecayRadius_,
          actuatorLineInfo->gaussDecayRadius_);

        // number of points for this line
        double numPoints = 10;
        get_if_present(y_spec, "number_of_points", numPoints, numPoints);
        actuatorLineInfo->numPoints_ = numPoints;

        // tip coordinates of this point
        const YAML::Node tipCoord = y_spec["tip_coordinates"];
        if (tipCoord)
          actuatorLineInfo->tipCoordinates_ = tipCoord.as<Coordinates>();
        else
          throw std::runtime_error(
            "ActuatorLinePointDrag: lacking tip_coordinates");

        // tail coordinates of this point
        const YAML::Node tailCoord = y_spec["tail_coordinates"];
        if (tailCoord)
          actuatorLineInfo->tailCoordinates_ = tailCoord.as<Coordinates>();
        else
          throw std::runtime_error(
            "ActuatorLinePointDrag: lacking tail_coordinates");
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
ActuatorLinePointDrag::setup()
{
  // objective: declare the part, register coordinates; must be before
  // populate_mesh()
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
ActuatorLinePointDrag::initialize()
{
  stk::mesh::BulkData& bulkData = realm_.bulk_data();

  // initialize need to ghost and elems to ghost
  needToGhostCount_ = 0;
  elemsToGhost_.clear();

  actuatorPointInfoMap_.clear();

  bulkData.modification_begin();

  if (actuatorGhosting_ == NULL) {
    // create new ghosting
    std::string theGhostName = "nalu_actuator_line_ghosting";
    actuatorGhosting_ = &bulkData.create_ghosting(theGhostName);
  } else {
    bulkData.destroy_ghosting(*actuatorGhosting_);
  }

  bulkData.modification_end();

  // clear some of the search info
  boundingSphereVec_.clear();
  boundingElementBoxVec_.clear();
  searchKeyPair_.clear();

  // set all of the candidate elements in the search target names
  populate_candidate_elements();

  // create the ActuatorLinePointDragPointInfo
  create_actuator_line_point_info_map();

  // coarse search
  determine_elems_to_ghost();

  // manage ghosting
  manage_ghosting();

  // complete filling in the set of elements connected to the centroid
  complete_search();
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ActuatorLinePointDrag::execute()
{
  // do we have mesh motion?
  if (actuatorLineMotion_)
    initialize();

  // meta/bulk data and nDim
  stk::mesh::MetaData& metaData = realm_.meta_data();
  stk::mesh::BulkData& bulkData = realm_.bulk_data();
  const int nDim = metaData.spatial_dimension();

  // extract fields
  VectorFieldType* coordinates = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name());
  VectorFieldType* velocity =
    metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  VectorFieldType* actuator_source = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "actuator_source");
  VectorFieldType* actuator_source_lhs = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "actuator_source_lhs");
  ScalarFieldType* density =
    metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  // deal with proper viscosity
  const std::string viscName =
    realm_.is_turbulent() ? "effective_viscosity" : "viscosity";
  ScalarFieldType* viscosity =
    metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, viscName);

  // fixed size scratch
  std::vector<double> ws_pointGasVelocity(nDim);
  std::vector<double> ws_elemCentroid(nDim);
  std::vector<double> ws_pointForce(nDim);
  std::vector<double> ws_elemDrag(nDim);
  double ws_pointGasDensity;
  double ws_pointGasViscosity;
  std::vector<double> ws_pointForceLHS(nDim);

  // zero out source term; do this manually since there are custom ghosted
  // entities
  stk::mesh::Selector s_nodes = stk::mesh::selectField(*actuator_source);
  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, s_nodes);
  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();
    double* actSrc = stk::mesh::field_data(*actuator_source, b);
    double* actSrcLhs = stk::mesh::field_data(*actuator_source_lhs, b);
    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      const int offSet = k * nDim;
      for (int j = 0; j < nDim; ++j) {
        actSrc[offSet + j] = 0.0;
        actSrcLhs[offSet + j] = 0.0;
      }
    }
  }

  // parallel communicate data to the ghosted elements; again can communicate
  // points to element ranks
  if (NULL != actuatorGhosting_) {
    std::vector<const stk::mesh::FieldBase*> ghostFieldVec;
    // fields that are needed
    ghostFieldVec.push_back(coordinates);
    ghostFieldVec.push_back(velocity);
    ghostFieldVec.push_back(viscosity);
    stk::mesh::communicate_field_data(*actuatorGhosting_, ghostFieldVec);
  }

  // loop over map and assemble source terms
  for (auto&& iterPoint : actuatorPointInfoMap_) {

    // actuator line info object of interest
    auto infoObject =
      dynamic_cast<ActuatorLinePointDragPointInfo*>(iterPoint.second.get());
    if (infoObject == NULL) {
      throw std::runtime_error(
        "Object in ActuatorPointInfo is not correct type.  Should be "
        "ActuatorLinePointDragPointInfo.");
    }

    //==========================================================================
    // extract the best element; compute drag given this velocity, property, etc
    // this point drag value will be used by all other elements below
    //==========================================================================
    stk::mesh::Entity bestElem = infoObject->bestElem_;
    int nodesPerElement = bulkData.num_nodes(bestElem);

    // resize some work vectors
    resize_std_vector(nDim, ws_coordinates_, bestElem, bulkData);
    resize_std_vector(nDim, ws_velocity_, bestElem, bulkData);
    resize_std_vector(1, ws_viscosity_, bestElem, bulkData);
    resize_std_vector(1, ws_density_, bestElem, bulkData);

    // gather nodal data to element nodes; both vector and scalar; coords are
    // used in determinant calc
    gather_field(
      nDim, &ws_coordinates_[0], *coordinates, bulkData.begin_nodes(bestElem),
      nodesPerElement);
    gather_field_for_interp(
      nDim, &ws_velocity_[0], *velocity, bulkData.begin_nodes(bestElem),
      nodesPerElement);
    gather_field_for_interp(
      1, &ws_viscosity_[0], *viscosity, bulkData.begin_nodes(bestElem),
      nodesPerElement);
    gather_field_for_interp(
      1, &ws_density_[0], *density, bulkData.begin_nodes(bestElem),
      nodesPerElement);

    // compute volume
    double bestElemVolume = compute_volume(nDim, bestElem, bulkData);

    // interpolate velocity
    interpolate_field(
      nDim, bestElem, bulkData, &(infoObject->isoParCoords_[0]),
      &ws_velocity_[0], &ws_pointGasVelocity[0]);

    // interpolate viscosity
    interpolate_field(
      1, bestElem, bulkData, &(infoObject->isoParCoords_[0]), &ws_viscosity_[0],
      &ws_pointGasViscosity);

    // interpolate density
    interpolate_field(
      1, bestElem, bulkData, &(infoObject->isoParCoords_[0]), &ws_density_[0],
      &ws_pointGasDensity);

    // point drag calculation
    compute_point_drag(
      nDim, infoObject->searchRadius_, &infoObject->velocity_[0],
      &ws_pointGasVelocity[0], ws_pointGasViscosity, ws_pointGasDensity,
      &ws_pointForce[0], &ws_pointForceLHS[0]);

    // assemble nodal lhs quantity for best elem
    assemble_lhs_to_best_elem_nodes(
      nDim, bestElem, bulkData, bestElemVolume, &ws_pointForceLHS[0],
      *actuator_source_lhs);

    // get the vector of elements
    const std::set<stk::mesh::Entity>& nodeVec = infoObject->nodeVec_;

    spread_actuator_force_to_node_vec(
      nDim, nodeVec, ws_pointForce, &(infoObject->centroidCoords_[0]),
      *coordinates, *actuator_source, infoObject->gaussDecayRadius_);
  }

  // parallel assemble (contributions from ghosted and locally owned)
  std::vector<const stk::mesh::FieldBase*> sumFieldVec;
  sumFieldVec.push_back(actuator_source);
  sumFieldVec.push_back(actuator_source_lhs);
  stk::mesh::parallel_sum_including_ghosts(bulkData, sumFieldVec);
}
//--------------------------------------------------------------------------
//-------- create_actuator_line_point_info_map -----------------------------
//--------------------------------------------------------------------------
void
ActuatorLinePointDrag::create_actuator_line_point_info_map()
{

  const double currentTime = realm_.get_current_time();

  stk::mesh::MetaData& metaData = realm_.meta_data();
  const int nDim = metaData.spatial_dimension();

  for (size_t k = 0; k < actuatorInfo_.size(); ++k) {

    const auto actuatorLineInfo =
      dynamic_cast<ActuatorLinePointDragInfo*>(actuatorInfo_[k].get());

    if (actuatorLineInfo == NULL) {
      throw std::runtime_error("Object in ActuatorInfo is not correct type.  "
                               "Should be ActuatorLineDragInfo.");
    }

    int processorId = actuatorLineInfo->processorId_;
    if (processorId == NaluEnv::self().parallel_rank()) {

      // define a point that will hold the centroid
      Point centroidCoords;

      // determine the distance between points and line centroid (for rotation)
      double dx[3] = {};
      double lineCentroid[3] = {};
      double velocity[3] = {};
      double currentCoords[3] = {};

      std::vector<double> tipC(nDim);
      tipC[0] = actuatorLineInfo->tipCoordinates_.x_;
      tipC[1] = actuatorLineInfo->tipCoordinates_.y_;

      std::vector<double> tailC(nDim);
      tailC[0] = actuatorLineInfo->tailCoordinates_.x_;
      tailC[1] = actuatorLineInfo->tailCoordinates_.y_;

      if (nDim > 2) {
        tipC[2] = actuatorLineInfo->tipCoordinates_.z_;
        tailC[2] = actuatorLineInfo->tailCoordinates_.z_;
      }

      const int numPoints = actuatorLineInfo->numPoints_;
      for (int j = 0; j < nDim; ++j) {
        dx[j] = (tipC[j] - tailC[j]) / (double)(numPoints - 1);
        lineCentroid[j] = (tipC[j] + tailC[j]) / 2.0;
      }

      // loop over all points
      for (int np = 0; np < numPoints; ++np) {
        // extract current localPointId; increment for next one up...
        size_t localPointId = localPointId_++;
        stk::search::IdentProc<uint64_t, int> theIdent(
          localPointId, NaluEnv::self().parallel_rank());

        // set model coordinates
        for (int j = 0; j < nDim; ++j)
          currentCoords[j] = tailC[j] + np * dx[j];

        // move the coordinates; set the velocity... may be better on the
        // lineInfo object
        set_current_coordinates(
          lineCentroid, currentCoords, actuatorLineInfo->omega_, currentTime);
        set_current_velocity(
          lineCentroid, currentCoords, velocity, actuatorLineInfo->omega_);

        for (int j = 0; j < nDim; ++j)
          centroidCoords[j] = currentCoords[j];

        // create the bounding point sphere and push back
        boundingSphere theSphere(
          Sphere(centroidCoords, actuatorLineInfo->radius_), theIdent);
        boundingSphereVec_.push_back(theSphere);

        // create the point info and push back to map
        actuatorPointInfoMap_.insert(std::make_pair(
          localPointId,
          make_unique<ActuatorLinePointDragPointInfo>(
            centroidCoords, actuatorLineInfo->radius_, actuatorLineInfo->omega_,
            actuatorLineInfo->gaussDecayRadius_, velocity)));
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- set_current_coordinates -----------------------------------------
//--------------------------------------------------------------------------
void
ActuatorLinePointDrag::set_current_coordinates(
  double* lineCentroid,
  double* centroidCoords,
  const double& omega,
  const double& currentTime)
{
  // hack for rotation only in the y-z plane
  const double cY = centroidCoords[1] - lineCentroid[1];
  const double cZ = centroidCoords[2] - lineCentroid[2];
  const double sinOT = sin(omega * currentTime);
  const double cosOT = cos(omega * currentTime);

  // no change to x-coordinates
  centroidCoords[1] = sinOT * cZ + cosOT * cY + lineCentroid[1];
  centroidCoords[2] = cosOT * cZ - sinOT * cY + lineCentroid[2];
}

//--------------------------------------------------------------------------
//-------- set_current_velocity --------------------------------------------
//--------------------------------------------------------------------------
void
ActuatorLinePointDrag::set_current_velocity(
  double* lineCentroid,
  const double* centroidCoords,
  double* velocity,
  const double& omega)
{
  // hack for rotation only in the y-z plane
  double cY = centroidCoords[1] - lineCentroid[1];
  double cZ = centroidCoords[2] - lineCentroid[2];

  velocity[0] = 0.0;
  velocity[1] = -omega * cZ;
  velocity[2] = +omega * cY;
}

//--------------------------------------------------------------------------
//-------- assemble_source_to_nodes ----------------------------------------
//-------------------------------------------------------------------------
void
ActuatorLinePointDrag::assemble_lhs_to_best_elem_nodes(
  const int& nDim,
  stk::mesh::Entity elem,
  const stk::mesh::BulkData& bulkData,
  const double& elemVolume,
  const double* dragLHS,
  stk::mesh::FieldBase& actuator_source_lhs)
{
  // extract master element from the bucket in which the element resides
  const stk::topology& elemTopo = bulkData.bucket(elem).topology();
  MasterElement* meSCV =
    sierra::nalu::MasterElementRepo::get_volume_master_element(elemTopo);
  const int numScvIp = meSCV->numIntPoints_;

  // extract elem_node_relations
  stk::mesh::Entity const* elem_node_rels = bulkData.begin_nodes(elem);

  // assemble to nodes
  const int* ipNodeMap = meSCV->ipNodeMap();
  for (int ip = 0; ip < numScvIp; ++ip) {

    // nearest node to ip
    const int nearestNode = ipNodeMap[ip];

    // extract node and pointer to source term
    stk::mesh::Entity node = elem_node_rels[nearestNode];
    double* sourceTermLHS =
      (double*)stk::mesh::field_data(actuator_source_lhs, node);

    // nodal weight based on volume weight
    const double nodalWeight = ws_scv_volume_[ip] / elemVolume;
    for (int j = 0; j < nDim; j++)
      sourceTermLHS[j] += nodalWeight * dragLHS[j];
  }
}

// Spread actuator force to nodes
void
ActuatorLinePointDrag::spread_actuator_force_to_node_vec(
  const int& nDim,
  const std::set<stk::mesh::Entity>& nodeVec,
  const std::vector<double>& actuator_force,
  const double* actuator_node_coordinates,
  const stk::mesh::FieldBase& coordinates,
  stk::mesh::FieldBase& actuator_source,
  const double& epsilon)
{
  std::vector<double> ws_nodeForce(nDim);
  // iterate over node vector, calculate and apply source term
  std::set<stk::mesh::Entity>::iterator iNode;
  for (iNode = nodeVec.begin(); iNode != nodeVec.end(); ++iNode) {

    stk::mesh::Entity node = *iNode;
    const double* node_coords =
      (double*)stk::mesh::field_data(coordinates, node);
    const double radius =
      compute_distance(nDim, node_coords, actuator_node_coordinates);
    // project the force to this node with projection function
    compute_node_drag_given_radius(
      nDim, radius, epsilon, &actuator_force[0], &ws_nodeForce[0]);
    double* sourceTerm = (double*)stk::mesh::field_data(actuator_source, node);
    for (int j = 0; j < nDim; ++j)
      sourceTerm[j] = ws_nodeForce[j];
  }
}

std::string
ActuatorLinePointDrag::get_class_name()
{
  return "ActuatorLinePointDrag";
}
} // namespace nalu
} // namespace sierra
