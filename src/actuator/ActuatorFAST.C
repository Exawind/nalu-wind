// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include <actuator/ActuatorFAST.h>
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

// The utilities used for actuator
#include "actuator/UtilitiesActuator.h"

namespace sierra {
namespace nalu {

// constructor
ActuatorFASTInfo::ActuatorFASTInfo() : ActuatorInfo(), fllt_correction_(false)
{}

// destructor
ActuatorFASTInfo::~ActuatorFASTInfo()
{
  // nothing to do
}

// constructor
ActuatorFASTPointInfo::ActuatorFASTPointInfo(
  size_t globTurbId,
  Point centroidCoords,
  double searchRadius,
  Coordinates epsilon,
  Coordinates epsilon_opt,
  fast::ActuatorNodeType nType,
  int forceInd)
  : ActuatorPointInfo(
      centroidCoords, searchRadius, 1.0e16, stk::mesh::Entity()),
    globTurbId_(globTurbId),
    epsilon_(epsilon),
    epsilon_opt_(epsilon_opt),
    nodeType_(nType),
    forcePntIndex_(forceInd)
{
  // nothing to do
}

// destructor
ActuatorFASTPointInfo::~ActuatorFASTPointInfo()
{
  // nothing to do
}

// constructor
ActuatorFAST::ActuatorFAST(Realm& realm, const YAML::Node& node)
  : Actuator(realm, node), numFastPoints_(0)
{
  // Actuator::load(node);
  // load the data
  load(node);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ActuatorFAST::~ActuatorFAST()
{
  FAST.end(); // Call destructors in FAST_cInterface
}

// Multiply the point force by the weight at this element location.
void
ActuatorFAST::compute_node_force_given_weight(
  const int& nDim, const double& g, const double* pointForce, double* nodeForce)
{

  for (int j = 0; j < nDim; ++j)
    nodeForce[j] = pointForce[j] * g;
}

//--------------------------------------------------------------------------
//-------- load ------------------------------------------------------------
//--------------------------------------------------------------------------
void
ActuatorFAST::load(const YAML::Node& y_node)
{
  // check for any data probes
  const YAML::Node y_actuator = y_node["actuator"];
  if (y_actuator) {
    // Populate object of inputs class to FAST
    fi.comm = NaluEnv::self().parallel_comm();

    get_required(y_actuator, "n_turbines_glob", fi.nTurbinesGlob);

    if (fi.nTurbinesGlob > 0) {

      get_if_present(y_actuator, "dry_run", fi.dryRun, false);
      get_if_present(y_actuator, "debug", fi.debug, false);
      get_required(y_actuator, "t_start", fi.tStart);
      std::string simStartType = "na";
      get_required(y_actuator, "simStart", simStartType);
      if (simStartType == "init") {
        if (fi.tStart == 0) {
          fi.simStart = fast::init;
        } else {
          throw std::runtime_error(
            get_class_name() +
            ": simStart type not consistent with start time for FAST");
        }
      } else if (simStartType == "trueRestart") {
        fi.simStart = fast::trueRestart;
      } else if (simStartType == "restartDriverInitFAST") {
        fi.simStart = fast::restartDriverInitFAST;
      }
      get_required(y_actuator, "n_every_checkpoint", fi.nEveryCheckPoint);
      get_required(y_actuator, "dt_fast", fi.dtFAST);
      get_required(
        y_actuator, "t_max",
        fi.tMax); // tMax is the total duration to which you want to run FAST.
      // This should be the same or greater than the max time given
      // in the FAST fst file. Choose this carefully as FAST writes
      // the output file only at this point if you choose the binary
      // file output.

      if (y_actuator["super_controller"]) {
        get_required(y_actuator, "super_controller", fi.scStatus);
        get_required(y_actuator, "sc_libFile", fi.scLibFile);
        get_required(y_actuator, "num_sc_inputs", fi.numScInputs);
        get_required(y_actuator, "num_sc_outputs", fi.numScOutputs);
      }

      fi.globTurbineData.resize(fi.nTurbinesGlob);

      for (int iTurb = 0; iTurb < fi.nTurbinesGlob; iTurb++) {
        if (y_actuator["Turbine" + std::to_string(iTurb)]) {

          const YAML::Node cur_turbine =
            y_actuator["Turbine" + std::to_string(iTurb)];

          actuatorInfo_.emplace_back(new ActuatorFASTInfo());

          auto actuatorFASTInfo =
            dynamic_cast<ActuatorFASTInfo*>(actuatorInfo_.back().get());

          get_required(
            cur_turbine, "turbine_name", actuatorFASTInfo->turbineName_);

          std::string turbFileName;
          get_if_present(cur_turbine, "file_to_dump_turb_pts", turbFileName);
          if (!turbFileName.empty()) {
            actuatorFASTInfo->fileToDumpPoints_ = turbFileName;
          }

          // The correction from filtered lifting line theory
          get_if_present_no_default(cur_turbine, "fllt_correction", actuatorFASTInfo->fllt_correction_);

          // The value epsilon / chord [non-dimensional]
          // This is a vector containing the values for:
          //   - chord aligned (x),
          //   - tangential to chord (y),
          //   - spanwise (z)
          const YAML::Node epsilon_chord = cur_turbine["epsilon_chord"];
          const YAML::Node epsilon = cur_turbine["epsilon"];
          if(epsilon && epsilon_chord){
            throw std::runtime_error("epsilon and epsilon_chord have both been specified for Turbine "
              + std::to_string(iTurb) + "\nYou must pick one or the other.");
          }
          if(epsilon && actuatorFASTInfo->fllt_correction_){
            throw std::runtime_error("epsilon and fllt_correction have both been specified for Turbine "
              +std::to_string(iTurb) + "\nepsilon_chord and epsilon_min should be used with fllt_correction.");
          }

          // If epsilon/chord is given, store it,
          // If it is not given, set it to zero, such
          // that it is smaller than the standard epsilon and
          // will not be used
          if ( epsilon_chord )
          {
            // epsilon / chord
            actuatorFASTInfo->epsilon_chord_ = epsilon_chord.as<Coordinates>();

            // Minimum epsilon allowed in simulation. This is required when
            //   specifying epsilon/chord
            get_required(cur_turbine, "epsilon_min",
              actuatorFASTInfo->epsilon_min_);
          }
          // Set all unused epsilon values to zero
          else
          {
            actuatorFASTInfo->epsilon_chord_.x_ = 0.;
            actuatorFASTInfo->epsilon_chord_.y_ = 0.;
            actuatorFASTInfo->epsilon_chord_.z_ = 0.;
            actuatorFASTInfo->epsilon_min_.x_ = 0.;
            actuatorFASTInfo->epsilon_min_.y_ = 0.;
            actuatorFASTInfo->epsilon_min_.z_ = 0.;
          }

          // Check if epsilon is given and store it.
          if ( epsilon ) {
            // Store the minimum epsilon
            actuatorFASTInfo->epsilon_ = epsilon.as<Coordinates>();
          }
          // If epsilon/chord is given and not standard epsilon, then assign
          //   the minimum epsilon as standard epsilon
          else  if (epsilon_chord) {
            // Get the minimum epsilon
            actuatorFASTInfo->epsilon_ = actuatorFASTInfo->epsilon_min_;
          }
          // If none of the conditions are met, throw an error
          else {
            throw std::runtime_error(
              "ActuatorLineFAST: lacking epsilon vector");
          }

          // An epsilon value used for the tower
          const YAML::Node epsilon_tower = cur_turbine["epsilon_tower"];
          // If epsilon tower is given store it.
          // If not, use the standard epsilon value
          if ( epsilon_tower )
            actuatorFASTInfo->epsilon_tower_ = epsilon_tower.as<Coordinates>();
          else
            actuatorFASTInfo->epsilon_tower_ = actuatorFASTInfo->epsilon_;

          readTurbineData(
            iTurb, fi, y_actuator["Turbine" + std::to_string(iTurb)]);
        } else {
          throw std::runtime_error(
            "Node for Turbine" + std::to_string(iTurb) +
            " not present in input file or I cannot read it");
        }
      }

    } else {
      throw std::runtime_error("Number of turbines <= 0 ");
    }
    FAST.setInputs(fi);
  }
}

void
ActuatorFAST::readTurbineData(
  int iTurb, fast::fastInputs& fi, YAML::Node turbNode)
{

  // Read turbine data for a given turbine using the YAML node
  get_required(turbNode, "turb_id", fi.globTurbineData[iTurb].TurbID);
  get_required(
    turbNode, "fast_input_filename",
    fi.globTurbineData[iTurb].FASTInputFileName);
  get_required(
    turbNode, "restart_filename",
    fi.globTurbineData[iTurb].FASTRestartFileName);
  if (turbNode["turbine_base_pos"].IsSequence()) {
    fi.globTurbineData[iTurb].TurbineBasePos =
      turbNode["turbine_base_pos"].as<std::vector<double>>();
  }
  if (turbNode["turbine_hub_pos"].IsSequence()) {
    fi.globTurbineData[iTurb].TurbineHubPos =
      turbNode["turbine_hub_pos"].as<std::vector<double>>();
  }
  get_required(
    turbNode, "num_force_pts_blade",
    fi.globTurbineData[iTurb].numForcePtsBlade);
  get_required(
    turbNode, "num_force_pts_tower", fi.globTurbineData[iTurb].numForcePtsTwr);
  fi.globTurbineData[iTurb].nacelle_cd = 0.0;
  fi.globTurbineData[iTurb].nacelle_area = 0.0;
  fi.globTurbineData[iTurb].air_density = 0.0;
  get_if_present(turbNode, "nacelle_cd", fi.globTurbineData[iTurb].nacelle_cd);
  get_if_present(
    turbNode, "nacelle_area", fi.globTurbineData[iTurb].nacelle_area);
  get_if_present(
    turbNode, "air_density", fi.globTurbineData[iTurb].air_density);
}

/** Called after load, but before initialize. The mesh isn't loaded yet. For
   now, this function only checks that the Nalu time step is an integral
   multiple of the FAST time step
*/
void
ActuatorFAST::setup()
{
  // objective: declare the part, register coordinates; must be before
  // populate_mesh()

  double dtNalu = realm_.get_time_step_from_file();
  tStepRatio_ = dtNalu / fi.dtFAST;
  if (std::abs(dtNalu - tStepRatio_ * fi.dtFAST) < 0.001) { // TODO: Fix
    // arbitrary number
    // 0.001
    NaluEnv::self().naluOutputP0()
        << "Time step ratio  dtNalu/dtFAST: " << tStepRatio_ << std::endl;
  } else {
    throw std::runtime_error("ActuatorFAST: Ratio of Nalu's time step is not "
                             "an integral multiple of FAST time step");
  }
}

/** This function searches for the processor containing the hub point of each
 * turbine and allocates the turbine to that processor. It does this through a
 * stk::coarse_search of bounding boxes around the processor domains.
 */
void
ActuatorFAST::allocateTurbinesToProcs()
{
  stk::mesh::MetaData& metaData = realm_.meta_data();

  // clear some of the search info
  boundingHubSphereVec_.clear();
  boundingProcBoxVec_.clear();
  searchKeyPair_.clear();

  const int nDim = metaData.spatial_dimension();

  // set all of the candidate elements in the search target names
  populate_candidate_procs();

  const size_t nTurbinesGlob = FAST.get_nTurbinesGlob();
  thrust.resize(nTurbinesGlob);
  torque.resize(nTurbinesGlob);
  for (size_t iTurb = 0; iTurb < nTurbinesGlob; ++iTurb) {
    thrust[iTurb].resize(nDim);
    torque[iTurb].resize(nDim);
  }

  for (size_t iTurb = 0; iTurb < nTurbinesGlob; ++iTurb) {

    theKey theIdent(
      NaluEnv::self().parallel_rank(), NaluEnv::self().parallel_rank());

    // define a point that will hold the hub location
    Point hubPointCoords;
    std::vector<double> hubCoords(3, 0.0);
    FAST.getApproxHubPos(hubCoords, iTurb);
    for (int j = 0; j < nDim; ++j)
      hubPointCoords[j] = hubCoords[j];
    boundingSphere theSphere(Sphere(hubPointCoords, 1.0), theIdent);
    boundingHubSphereVec_.push_back(theSphere);
  }
  stk::search::coarse_search(
    boundingHubSphereVec_, boundingProcBoxVec_, searchMethod_,
    NaluEnv::self().parallel_comm(), searchKeyPair_, false);

  int iTurb = 0;
  std::vector<
  std::pair<boundingSphere::second_type, boundingElementBox::second_type>>::
      const_iterator ii;
  for (ii = searchKeyPair_.begin(); ii != searchKeyPair_.end(); ++ii) {
    const unsigned box_proc = ii->second.proc();

    FAST.setTurbineProcNo(iTurb, box_proc);
    iTurb++;
  }
}

/** This method allocates the turbines to processors, initializes the OpenFAST
 * instances of each turbine, populates the map of ActuatorLinePointInfo with
 * the actuator points of all turbines and determines the elements associated
 * with each actuator point.
 */
void
ActuatorFAST::initialize()
{

  allocateTurbinesToProcs();

  FAST.init();

  //
  // This is done to create the actuator point info map once
  //
  stk::mesh::BulkData& bulkData = realm_.bulk_data();

  // initialize need to ghost and elems to ghost
  needToGhostCount_ = 0;
  elemsToGhost_.clear();

  // clear actuatorPointInfoMap_
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

  // create the ActuatorLineFASTPointInfo
  create_actuator_point_info_map();

  create_point_info_map_class_specific();

  // coarse search
  determine_elems_to_ghost();

  // manage ghosting
  manage_ghosting();

  // complete filling in the set of elements connected to the centroid
  complete_search();

}

/**
 * This method should be called whenever the actuator points have moved and does
 * the following:
 *
 * + creates a new map of actuator points in ActuatorLinePointInfoMap,
 * + searches the element bounding boxes for the elements within the search
 * radius of each actuator point,
 * + identifies the elements to be ghosted to the processor controlling the
 * turbine,
 * + identifies the bestElem_ that contains each actuator point.
 */
void
ActuatorFAST::update()
{
  stk::mesh::BulkData& bulkData = realm_.bulk_data();

  // initialize need to ghost and elems to ghost
  needToGhostCount_ = 0;
  elemsToGhost_.clear();

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

  // create the ActuatorLineFASTPointInfo
  update_actuator_point_info_map();

  // coarse search
  determine_elems_to_ghost();

  // manage ghosting
  manage_ghosting();

  // complete filling in the set of elements connected to the centroid
  complete_search();

}

/** This function is called at each time step. This samples the velocity at each
 * actuator point, advances the OpenFAST turbine models to Nalu's next time step
 * and assembles the source terms in the momentum equation for Nalu.
 */
void
ActuatorFAST::execute()
{
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
  ScalarFieldType* actuator_source_lhs = metaData.get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "actuator_source_lhs");
  ScalarFieldType* g =
    metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "g");
  ScalarFieldType* density =
    metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  ScalarFieldType* dualNodalVolume = metaData.get_field<ScalarFieldType>(
                                       stk::topology::NODE_RANK, "dual_nodal_volume");
  // deal with proper viscosity
  //  const std::string viscName = realm_.is_turbulent() ? "effective_viscosity"
  //  : "viscosity"; ScalarFieldType *viscosity
  //    = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK,
  //    viscName);

  // fixed size scratch
  std::vector<double> ws_pointGasVelocity(nDim);
  std::vector<double> ws_elemCentroid(nDim);
  std::vector<double> ws_nodeForce(nDim);
  double ws_pointGasDensity;
  //  double ws_pointGasViscosity;

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
    double* gF = stk::mesh::field_data(*g, b);
    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      actSrcLhs[k] = 0.0;
      gF[k] = 0.0;
      const int offSet = k * nDim;
      for (int j = 0; j < nDim; ++j) {
        actSrc[offSet + j] = 0.0;
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
    ghostFieldVec.push_back(dualNodalVolume);
    //    ghostFieldVec.push_back(viscosity);
    stk::mesh::communicate_field_data(*actuatorGhosting_, ghostFieldVec);
  }

  // loop over map and get velocity at points
  for (std::size_t np = 0; np < numFastPoints_; np++) {

    // actuator line info object of interest
    auto infoObject =
      dynamic_cast<ActuatorFASTPointInfo*>(actuatorPointInfoMap_.at(np).get());
    if (infoObject == NULL) {
      throw std::runtime_error("Object in ActuatorPointInfo is not the correct "
                               "type.  Should be ActuatorFASTPointInfo.");
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
    //    resize_std_vector(1, ws_viscosity_, bestElem, bulkData);
    resize_std_vector(1, ws_density_, bestElem, bulkData);

    // gather nodal data to element nodes; both vector and scalar; coords are
    // used in determinant calc
    gather_field(
      nDim, &ws_coordinates_[0], *coordinates, bulkData.begin_nodes(bestElem),
      nodesPerElement);
    gather_field_for_interp(
      nDim, &ws_velocity_[0], *velocity, bulkData.begin_nodes(bestElem),
      nodesPerElement);
    //    gather_field_for_interp(1, &ws_viscosity_[0], *viscosity,
    //    bulkData.begin_nodes(bestElem),
    //                            nodesPerElement);
    gather_field_for_interp(
      1, &ws_density_[0], *density, bulkData.begin_nodes(bestElem),
      nodesPerElement);

    // interpolate velocity
    interpolate_field(
      nDim, bestElem, bulkData, infoObject->isoParCoords_.data(),
      &ws_velocity_[0], ws_pointGasVelocity.data());

    // interpolate density
    interpolate_field(
      1, bestElem, bulkData, infoObject->isoParCoords_.data(), &ws_density_[0],
      &ws_pointGasDensity);
    int nNp = (int)np;
    
    /////////////////////////
    // Add the filtered lifting line theory correction here
    // This adds an extra component of velocity in every direction
    /////////////////////////
    for (int i=0; i<nDim; i++) {
      ws_pointGasVelocity.data()[i] += infoObject -> du.data()[i];
    }

    // Set the CFD velocity at the actuator node
    FAST.setVelocityForceNode(
    ws_pointGasVelocity, nNp, infoObject->globTurbId_);

  }
 
  // Add the filtered lifting line correction
  filtered_lifting_line();

  if (!FAST.isDryRun()) {

    FAST.interpolateVel_ForceToVelNodes();

    if (FAST.isTimeZero()) {
      FAST.solution0();
    }

    // move for acuatorLines, do nothing for actuatorDisks
    update_class_specific();

    // Step FAST
    for (int j = 0; j < tStepRatio_; j++)
      FAST.step();
  }

  // reset the thrust and torque from each turbine to zero
  const size_t nTurbinesGlob = FAST.get_nTurbinesGlob();
  for (size_t iTurb = 0; iTurb < nTurbinesGlob; iTurb++) {
    for (int j = 0; j < nDim; j++) {
      torque[iTurb][j] = 0.0;
      thrust[iTurb][j] = 0.0;
    }
  }

  // if disk average azimuthally
  // apply the forcing terms
  execute_class_specific(nDim, coordinates, actuator_source, dualNodalVolume);

  if (FAST.isDebug()) {
    for (size_t iTurb = 0; iTurb < nTurbinesGlob; iTurb++) {
      NaluEnv::self().naluOutput()
          << "  Thrust[" << iTurb << "] = " << thrust[iTurb][0] << " "
          << thrust[iTurb][1] << " " << thrust[iTurb][2] << " " << std::endl;
      NaluEnv::self().naluOutput()
          << "  Torque[" << iTurb << "] = " << torque[iTurb][0] << " "
          << torque[iTurb][1] << " " << torque[iTurb][2] << " " << std::endl;

      int processorId = FAST.get_procNo(iTurb);
      if (NaluEnv::self().parallel_rank() == processorId) {
        std::vector<double> tmpThrust(3);
        std::vector<double> tmpTorque(3);

        FAST.computeTorqueThrust(iTurb, tmpTorque, tmpThrust);

        NaluEnv::self().naluOutput()
            << "  Thrust ratio actual/correct = ["
            << thrust[iTurb][0] / tmpThrust[0] << " "
            << thrust[iTurb][1] / tmpThrust[1] << " "
            << thrust[iTurb][2] / tmpThrust[2] << "] " << std::endl;
        NaluEnv::self().naluOutput()
            << "  Torque ratio actual/correct = ["
            << torque[iTurb][0] / tmpTorque[0] << " "
            << torque[iTurb][1] / tmpTorque[1] << " "
            << torque[iTurb][2] / tmpTorque[2] << "] " << std::endl;
      }
    }
  }

  // parallel assemble (contributions from ghosted and locally owned)
  const std::vector<const stk::mesh::FieldBase*> sumFieldVec(
    1, actuator_source);
  stk::mesh::parallel_sum_including_ghosts(bulkData, sumFieldVec);

  const std::vector<const stk::mesh::FieldBase*> sumFieldG(1, g);
  stk::mesh::parallel_sum_including_ghosts(bulkData, sumFieldG);
}


// Copmute the filtered lifting line theory correction
void ActuatorFAST::filtered_lifting_line()
{


  // The number of dimensions (assumes 3D)
  int nDim=3;

  // The total number of turbines
  const size_t numTurbines = fi.nTurbinesGlob;

  // The index of actuator point
  // This identifiex an actuator point in a vector containing all actuator
  // points in the simulation
  size_t np;


  // Loop through all turbines
  for (size_t iTurb=0; iTurb < numTurbines; iTurb++ )
  {

    // Only use if correction is active for this turbine
    // Point to the last element of this array that was just appended
    auto actuatorFASTInfo = dynamic_cast<ActuatorFASTInfo*>(
      actuatorInfo_.at(iTurb).get());

    // If the correciton is not active for this turbine, skip this turbine
    if ( ! actuatorFASTInfo -> fllt_correction_) continue;

    // Do not consider this unless the turbine lies in the same processor
    if (FAST.get_procNo(iTurb) != NaluEnv::self().parallel_rank()) continue;

    // Number of blades
    const size_t numBlades = FAST.get_numBlades(iTurb);

    // The total number of actuator points per blade
    const size_t ptsPerBlade = FAST.get_numForcePtsBlade(iTurb); //totalActuatorNodes / numBlades;

    ///////////////////////////////
    // Step 1: Compute function G
    ///////////////////////////////

    // The force at the given actuator point
    std::vector<double> force(nDim);
    // The velocity at a given actuator point
    std::vector<double> vel(nDim);

    // Loop through all blade points
    for (size_t nb=0; nb < numBlades; nb++)
      {
      // Loop through all blade points
      for (size_t na=0; na < ptsPerBlade; na++)
        {

          // The actuator point index
          np = indexMap_[iTurb][nb][na];

          // actuator line info object of interest
          auto infoObject =
            dynamic_cast<ActuatorFASTPointInfo*>(
              actuatorPointInfoMap_.at(np).get());
        
          // Get the force from FAST
          FAST.getForce(force, np, infoObject->globTurbId_);
          // Get the velocity from FAST
          FAST.getRelativeVelForceNode(vel, np, infoObject->globTurbId_);

          // The velocity magnitude squared
          double vmag2(0);
          // Compute the dot product of the velocity (vmag^2)
          for (int i = 0; i < nDim; i++) vmag2 += vel[i] * vel[i];

          // The dot product of velocity and force
          double fvel(0);
          // Compute the dot product of the velocity (vmag^2)
          for (int i = 0; i < nDim; i++) fvel += force[i] * vel[i];

          // Compute the dr
          // This is the spanwise width
          double dr(0);

          // The coordinate location
          std::vector<double> xyz{0., 0., 0.};
          // The plus 1 coordinate
          std::vector<double> xyz_p1{0., 0., 0.};
          // The minus one coordinate
          std::vector<double> xyz_m1{0., 0., 0.};

          // Compute the radial difference between points
          if (na == 0)
          {
            // Get the adjacent index
            np = indexMap_[iTurb][nb][na];
            // Get the coordinates for the current point
            FAST.getForceNodeCoordinates(xyz, np, iTurb);

            // Get the adjacent index
            np = indexMap_[iTurb][nb][na + 1];
            // Get the coordinates for the current point
            FAST.getForceNodeCoordinates(xyz_p1, np, iTurb);

            // Compute the magnitude of the difference
            for (int i = 0; i < nDim; i++)
                dr += std::pow(xyz_p1.data()[i]-xyz.data()[i], 2);

            // Take the square root and divide by 2 (central difference)
            dr = std::sqrt(dr) / 2.;
          }

          else if (na == ptsPerBlade - 1)
          {
            // Get the adjacent index
            np = indexMap_[iTurb][nb][na];
            // Get the coordinates for the current point
            FAST.getForceNodeCoordinates(xyz, np, iTurb);

            // Get the adjacent index
            np = indexMap_[iTurb][nb][na - 1];
            // Get the coordinates for the current point
            FAST.getForceNodeCoordinates(xyz_m1, np, iTurb);

            // Compute the magnitude of the difference
            for (int i = 0; i < nDim; i++)
                dr += std::pow(xyz.data()[i]-xyz_m1.data()[i], 2);

            // Take the square root and divide by 2 (central difference)
            dr = std::sqrt(dr) / 2.;
          }

          else
          {
            // Get the adjacent index
            np = indexMap_[iTurb][nb][na-1];
            // Get the coordinates for the adjacent points i-1 and i+1
            FAST.getForceNodeCoordinates(xyz_m1, np, iTurb);
  
            // Get the adjacent index
            np = indexMap_[iTurb][nb][na+1];
            // Get the coordinates for the adjacent points i-1 and i+1
            FAST.getForceNodeCoordinates(xyz_p1, np, iTurb);
 
            // Compute the magnitude of the difference
            for (int i = 0; i < nDim; i++)
                dr += std::pow(xyz_p1.data()[i]-xyz_m1.data()[i], 2);

            // Take the square root and divide by 2 (central difference)
            dr = std::sqrt(dr) / 2.;
          }
              

          // Compute the function G
          // This is the same as the lift vector along the blade span
          for (int i = 0; i < nDim; i++)
          {
            // Compute the vector G
            infoObject -> G.data()[i] = force[i] - vel[i] * fvel / vmag2;

            // Convert G to force per unit width
            infoObject -> G.data()[i] /= dr;

            // Zero the induced velocity values
            infoObject -> u_LES.data()[i] = 0;
            infoObject -> u_opt.data()[i] = 0;
          }
        }
      }

    ///////////////////////////////
    // Step 2: Compute gradient of G
    ///////////////////////////////

    // Loop through all blades
    for (size_t nb=0; nb < numBlades; nb++)
      {

      // The first actuator point
      np = indexMap_[iTurb][nb][0];
      auto infoObject_0 =
          dynamic_cast<ActuatorFASTPointInfo*>(
            actuatorPointInfoMap_.at(np).get());
      // The last actuator point
      np = indexMap_[iTurb][nb][ptsPerBlade-1];
      auto infoObject_N =
          dynamic_cast<ActuatorFASTPointInfo*>(
            actuatorPointInfoMap_.at(np).get());

      // The gradient of the first and last points
      for (int i = 0; i < nDim; i++) {
        infoObject_0 -> dG.data()[i] =  infoObject_0 -> G.data()[i];
        infoObject_N -> dG.data()[i] = -infoObject_N -> G.data()[i];
      }

      // Loop through all blade points
      for (size_t na=1; na < ptsPerBlade-1; na++)
        {
          // The actuator point index
          np = indexMap_[iTurb][nb][na];
          // Index before np - 1
          size_t np_m1 = indexMap_[iTurb][nb][na-1];
          // Index after np + 1
          size_t np_p1 = indexMap_[iTurb][nb][na+1];

          // actuator line info object of interest
          // This is where dG is being computed
          auto infoObject =
              dynamic_cast<ActuatorFASTPointInfo*>(
                actuatorPointInfoMap_.at(np).get());
          auto infoObject_m1 =
              dynamic_cast<ActuatorFASTPointInfo*>(
                actuatorPointInfoMap_.at(np_m1).get());
          auto infoObject_p1 =
              dynamic_cast<ActuatorFASTPointInfo*>(
                actuatorPointInfoMap_.at(np_p1).get());

        // Compute the gradient using central differencing
        for (int i = 0; i < nDim; i++) {
          // Central differencing
          infoObject -> dG.data()[i] = (infoObject_p1 -> G.data()[i] - 
            infoObject_m1 -> G.data()[i]) / 2.;
          }
        }        
      }
  
  
    ///////////////////////////////
    // Step 3: Compute the induced velocities
    ///////////////////////////////

    // Loop through all blades
    for (size_t nb=0; nb < numBlades; nb++)
    {
      // Loop through all blade points
      for (int na=0; na < static_cast<int>(ptsPerBlade); na++)
      {
        // The actuator point index
        np = indexMap_[iTurb][nb][na];
        // The actuator point object
        auto infoObject =
            dynamic_cast<ActuatorFASTPointInfo*>(
              actuatorPointInfoMap_.at(np).get());

        // The coordinate of the actuator point
        const Point& r1= infoObject -> centroidCoords_;

        // Loop through all blade points
        for (int na_2=0; na_2 < static_cast<int>(ptsPerBlade); na_2++)
        {

          // Do not compute if at the same actuator point
          // In this case, the induction is zero
          if (na == na_2) continue;

          // The actuator point index
          size_t np_2 = indexMap_[iTurb][nb][na_2];

          // The actuator point object
          auto infoObject2 =
            dynamic_cast<ActuatorFASTPointInfo*>(
              actuatorPointInfoMap_.at(np_2).get());

          // The coordinate of the actuator point
          const Point& r2 = infoObject2 -> centroidCoords_;

          // Initialize the square of the difference to zero
          double rdiff2(0);
          // Compute the difference in radial location
          for (int i = 0; i < nDim; i++) 
          {
              rdiff2 += (r2[i] - r1[i]) * (r2[i] - r1[i]);
          }
          // The square root of this gives the magnitude of the vector
          double diff = std::sqrt(rdiff2);
          // Change the sign depending on which side the actuator point is on
          if (na_2 < na) diff *= -1;

          // Get the relative velocity
          // Get the velocity from FAST
          FAST.getRelativeVelForceNode(vel, np_2, infoObject->globTurbId_);        
          // The velocity magnitude squared
          double vmag(0);
          // Compute the dot product of the velocity (vmag^2)
          for (int i = 0; i < nDim; i++) vmag += vel[i] * vel[i];
          vmag = std::sqrt(vmag);

          // This is the gradient of the function G (it is a 3d vector)
          const std::array<double, 3>& dG = infoObject2 -> dG;

          // The value of epsilon
          // Notice the correciton assumes uniform epsilon and takes the 
          //   the first value
          const double& eps_les = infoObject -> epsilon_.x_;
          const double& eps_opt = infoObject -> epsilon_opt_.x_;

          // Compute the LES and optimal induced velocities
          for (int i = 0; i < nDim; i++) 
          {
              
            // Compute the LES induced velocity
            infoObject -> u_LES.data()[i] -= 1./(vmag * 4. * M_PI) * dG.data()[i] *
              (1. - std::exp(-rdiff2/(eps_les * eps_les))) / diff;

            // Compute the optimal induced velocity
            infoObject -> u_opt.data()[i] -= 1./(vmag * 4. * M_PI) * dG.data()[i] *
              (1. - std::exp(-rdiff2/(eps_opt * eps_opt))) / diff;

          }
        }
      }
    }

    ///////////////////////////////
    // Step 4: Compute the diifference in induced velocities
    ///////////////////////////////

    // The relaxation factor used in filtered lifting line theory
    double f = 0.01;

    // Loop through all blades
    for (size_t nb=0; nb < numBlades; nb++)
    {
      // Loop through all blade points
      for (size_t na=0; na < ptsPerBlade; na++)
      {

        // The actuator point index
        np = indexMap_[iTurb][nb][na];
        // The actuator point object
        auto infoObject =
            dynamic_cast<ActuatorFASTPointInfo*>(
              actuatorPointInfoMap_.at(np).get());

        // Loop through all directions
        for (int i=0; i<nDim; i++) {

          // Compute the difference between the velocity from the 
          //   les and the optimal
          infoObject -> du.data()[i] = infoObject -> du.data()[i] * (1.-f) 
            + f * (infoObject -> u_opt.data()[i]
              - infoObject -> u_LES.data()[i]);    
        }
      }
    }
  }  
}

// Creates bounding boxes around the subdomain of each processor
void
ActuatorFAST::populate_candidate_procs()
{
  stk::mesh::MetaData& metaData = realm_.meta_data();

  const int nDim = metaData.spatial_dimension();

  // fields
  VectorFieldType* coordinates = metaData.get_field<VectorFieldType>(
                                   stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // point data structures
  //  Point minCorner, maxCorner;
  std::vector<Point> minCorner(1), maxCorner(1);
  std::vector<Point> gMinCorner, gMaxCorner;

  // initialize max and min
  for (int j = 0; j < nDim; ++j) {
    minCorner[0][j] = +1.0e16;
    maxCorner[0][j] = -1.0e16;
  }

  // extract part
  stk::mesh::PartVector searchParts;
  for (size_t k = 0; k < searchTargetNames_.size(); ++k) {
    stk::mesh::Part* thePart = metaData.get_part(searchTargetNames_[k]);
    if (NULL != thePart)
      searchParts.push_back(thePart);
    else
      throw std::runtime_error(
        "ActuatorFAST: Part is null" + searchTargetNames_[k]);
  }

  // selector and bucket loop
  stk::mesh::Selector s_locally_owned =
    metaData.locally_owned_part() & stk::mesh::selectUnion(searchParts);

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, s_locally_owned);

  for (stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
       ib != elem_buckets.end(); ++ib) {

    stk::mesh::Bucket& b = **ib;

    const stk::mesh::Bucket::size_type length = b.size();

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {

      // get element
      stk::mesh::Entity node = b[k];

      // pointers to real data
      const double* coords = stk::mesh::field_data(*coordinates, node);

      // check max/min
      for (int j = 0; j < nDim; ++j) {
        minCorner[0][j] = std::min(minCorner[0][j], coords[j]);
        maxCorner[0][j] = std::max(maxCorner[0][j], coords[j]);
      }
    }
  }

  stk::parallel_vector_concat(
    NaluEnv::self().parallel_comm(), minCorner, gMinCorner);
  stk::parallel_vector_concat(
    NaluEnv::self().parallel_comm(), maxCorner, gMaxCorner);

  for (int j = 0; j < NaluEnv::self().parallel_size(); j++) {
    // setup ident
    stk::search::IdentProc<uint64_t, int> theIdent(j, j);

    // create the bounding point box and push back
    boundingElementBox theBox(Box(gMinCorner[j], gMaxCorner[j]), theIdent);
    boundingProcBoxVec_.push_back(theBox);
  }
}

//--------------------------------------------------------------------------
//-------- create_actuator_line_point_info_map -----------------------------
//--------------------------------------------------------------------------
void
ActuatorFAST::create_actuator_point_info_map()
{

  stk::mesh::MetaData& metaData = realm_.meta_data();
  const int nDim = metaData.spatial_dimension();

  size_t np = 0;

  for (size_t iTurb = 0; iTurb < actuatorInfo_.size(); ++iTurb) {

    const auto actuatorInfo =
      dynamic_cast<ActuatorFASTInfo*>(actuatorInfo_[iTurb].get());
    if (actuatorInfo == NULL) {
      throw std::runtime_error("Object in ActuatorInfo is not the correct "
                               "type.  It should be ActuatorFASTInfo.");
    }

    int processorId = FAST.get_procNo(iTurb);
    if (processorId == NaluEnv::self().parallel_rank()) {

      // define a point that will hold the centroid
      Point centroidCoords;

      // scratch array for coordinates and dummy array for velocity
      std::vector<double> currentCoords(3, 0.0);

      // loop over all points for this turbine
      const int numForcePts =
        FAST.get_numForcePts(iTurb); // Total number of elements

      if (!FAST.isDryRun()) {
        for (int iNode = 0; iNode < numForcePts; iNode++) {
          stk::search::IdentProc<uint64_t, int> theIdent(
            np, NaluEnv::self().parallel_rank());

          // set model coordinates from FAST
          // move the coordinates; set the velocity... may be better on the
          // lineInfo object
          FAST.getForceNodeCoordinates(currentCoords, np, iTurb);

          // Get the chord from inside of FAST to compute epsilon
          double chord = FAST.getChord(np, iTurb);

          // create the point info and push back to map
          Coordinates epsilon;
          // This is the optimal epsilon
          Coordinates epsilon_opt;
          
          // Go through all cases depending on what kind of actuator point it is
          switch (FAST.getForceNodeType(iTurb, np)) {

          case fast::HUB: {

            // The drag coefficient from the nacelle
            float nac_cd = FAST.get_nacelleCd(iTurb);

            // Compute epsilon only if drag coefficient is greater than zero
            if (nac_cd > 0) {

              // Calculate epsilon for hub node based on cd and area here
              float nac_area = FAST.get_nacelleArea(iTurb);

              // This model is used to set the momentum thickness
              // of the wake (Martinez-Tossas PhD Thesis 2017)
              float tmpEps = std::sqrt(2.0 / M_PI * nac_cd * nac_area);
              epsilon.x_ = tmpEps;
              epsilon.y_ = tmpEps;
              epsilon.z_ = tmpEps;
            }

            // If no nacelle force just specify the standard value
            // (it will not be used)
            else {
              epsilon = actuatorInfo->epsilon_;
            }
            
            epsilon_opt = epsilon;

            break;

            }

          // The epsilon along each blade point
          case fast::BLADE:

            // Define the optimal epsilon
            epsilon_opt.x_ = actuatorInfo->epsilon_chord_.x_ * chord;
            epsilon_opt.y_ = actuatorInfo->epsilon_chord_.y_ * chord;
            epsilon_opt.z_ = actuatorInfo->epsilon_chord_.z_ * chord;

            // Use epsilon based on the maximum between
            //   epsilon
            //   optimal epsilon (epsilon/chord)
            //   and the minimum epsilon because of grid resolution)
            // x direction
            epsilon.x_ = std::max(
                           std::max(
                               epsilon_opt.x_,
                               actuatorInfo->epsilon_min_.x_),
                               actuatorInfo->epsilon_.x_);
            // y direction
            epsilon.y_ = std::max(
                           std::max(
                               epsilon_opt.y_,
                               actuatorInfo->epsilon_min_.y_),
                               actuatorInfo->epsilon_.y_);

            // z direction
            epsilon.z_ = std::max(
                           std::max(
                               epsilon_opt.z_,
                               actuatorInfo->epsilon_min_.z_),
                               actuatorInfo->epsilon_.z_);

            break;

          // The value of epsilon related to the grid resolution
          case fast::TOWER:
            epsilon = actuatorInfo->epsilon_tower_;
            epsilon_opt = epsilon;
            break;

          // If no case, throw an error
          default:

            throw std::runtime_error("Actuator line model node type not valid");

            break;

          }

          // The radius of the searching. This is given in terms of 
          //   the maximum of epsilon.x/y/z/.
          double searchRadius = 
            std::max(epsilon.x_, std::max(epsilon.y_, epsilon.z_))
              * sqrt(log(1.0/0.001));

          for (int j = 0; j < nDim; ++j)
            centroidCoords[j] = currentCoords[j];

          // create the bounding point sphere and push back
          boundingSphere theSphere(
            Sphere(centroidCoords, searchRadius), theIdent);
          boundingSphereVec_.push_back(theSphere);

          // Insert all the information related to this actuator point
          // This is where the actuator point object is created
          actuatorPointInfoMap_.insert(std::make_pair(
                         np, make_unique<ActuatorFASTPointInfo>(
                           iTurb, 
                           centroidCoords, 
                           searchRadius, 
                           epsilon,
                           epsilon_opt,
                           FAST.getForceNodeType(iTurb, np), iNode)));

#if 0
          // Print the value of epsilon to the screen
          NaluEnv::self().naluOutput()
            << "  The value of epsilon for actuator point "
            << iNode
            << " is "
            << epsilon.x_ << "[m] "
            << epsilon.y_ << "[m] "
            << epsilon.z_ << "[m] "
            << " " << std::endl;
#endif

          // Counter for the number of blade points
          np = np + 1;
        }

      } else {
        NaluEnv::self().naluOutput()
            << "Proc " << NaluEnv::self().parallel_rank() << " glob iTurb "
            << iTurb << std::endl;
      }
    }
  }
  numFastPoints_ = actuatorPointInfoMap_.size();

  // execute this outside of loop so actuatorPoints that go into fast are
  // contiguous in the vectors (i.e. the first FAST.get_numForcePts() number of
  // points are the actuator lines, and whatever comes after them are the swept
  // points)
  create_point_info_map_class_specific();

  index_map();
  
}

//--------------------------------------------------------------------------
//-------- update_actuator_line_point_info_map -----------------------------
//--------------------------------------------------------------------------
void
ActuatorFAST::update_actuator_point_info_map()
{

  stk::mesh::MetaData& metaData = realm_.meta_data();
  const int nDim = metaData.spatial_dimension();

  size_t np = 0;

  for (size_t iTurb = 0; iTurb < actuatorInfo_.size(); ++iTurb) {

    const auto actuatorInfo =
      dynamic_cast<ActuatorFASTInfo*>(actuatorInfo_[iTurb].get());
    if (actuatorInfo == NULL) {
      throw std::runtime_error("Object in ActuatorInfo is not the correct "
                               "type.  It should be ActuatorFASTInfo.");
    }

    int processorId = FAST.get_procNo(iTurb);
    if (processorId == NaluEnv::self().parallel_rank()) {

      // define a point that will hold the centroid
      Point centroidCoords;

      // scratch array for coordinates and dummy array for velocity
      std::vector<double> currentCoords(3, 0.0);

      // loop over all points for this turbine
      const int numForcePts =
        FAST.get_numForcePts(iTurb); // Total number of elements

      if (!FAST.isDryRun()) {
        for (int iNode = 0; iNode < numForcePts; iNode++) {
          stk::search::IdentProc<uint64_t, int> theIdent(
            np, NaluEnv::self().parallel_rank());

          // set model coordinates from FAST
          // move the coordinates; set the velocity... may be better on the
          // lineInfo object
          FAST.getForceNodeCoordinates(currentCoords, np, iTurb);

          // Get the actuator point information
          auto infoObject = dynamic_cast<ActuatorFASTPointInfo*>(
              actuatorPointInfoMap_.at(np).get()); 

          // Clear the vector list
          infoObject -> nodeVec_.clear();
          infoObject -> bestX_ = 1.0e16;

          // Update the current coordinate
          for (int j = 0; j < nDim; ++j) {
            infoObject->centroidCoords_[j] = currentCoords[j];
            centroidCoords[j] = currentCoords[j];
          }

          // create the bounding point sphere and push back
          boundingSphere theSphere(
            Sphere(centroidCoords, infoObject->searchRadius_), 
              theIdent);
          boundingSphereVec_.push_back(theSphere);

          // Counter for the number of blade points
          np = np + 1;
        }

      } else {
        NaluEnv::self().naluOutput()
            << "Proc " << NaluEnv::self().parallel_rank() << " glob iTurb "
            << iTurb << std::endl;
      }
    }
  }
}

/// This function computes the index map such that actuator points can be
///   accessed using indexing:
///   (turbine number, blade number, actuator point number)
void ActuatorFAST::index_map()
{

  //////////////////////////////////////////////////////////////////////////////
  // Loop and map the indices
  // This creates a mapping which allows you to access the actuator point index
  //   based on turbine number, blade number, and actuator point number
  // Number of turbines
  int nt = fi.nTurbinesGlob;
  // Number of blades
  int nb = 0;
  // Number of actuator points
  int na = 0;
  // Loop through all turbines and only call FAST functions if processor owns 
  //   the turbine
  for (int i = 0; i < nt; ++i) {
    if (FAST.get_procNo(i) != NaluEnv::self().parallel_rank()) continue;
    nb = std::max(nb, FAST.get_numBlades(i));
    na = std::max(na, fi.globTurbineData[i].numForcePtsBlade);
  }

  // This is a map that stores the actuator point number np
  //   indexed using turbine number, blade number, actuator point number
  // Resize the original array to have
  //   (number of turbines, number of blades, number of actuator points)
  indexMap_.resize(nt);

  for (int i = 0; i<nt; ++i) {
      indexMap_[i].resize(nb);
    for (int j = 0; j<nb; ++j) {
      indexMap_[i][j].resize(na);        
    }
  }

  int it = -1;  // turbine number counter
  int ib = -1;  // blade number counter
  int actPtrCounter = -1;  // actuator point number counter


  // Loop through all actuator points and populate the index map
  for (int np=0; np < static_cast<int>(numFastPoints_); ++np) {

    // actuator line info object of interest
    auto infoObject =
      dynamic_cast<ActuatorFASTPointInfo*>(actuatorPointInfoMap_.at(np).get());

    // Only process blade actuator points
    if (infoObject->nodeType_ != fast::BLADE) continue;

    // This is the number of the specific turbine
    const int iTurb = static_cast<int>(infoObject->globTurbId_);

// Do not consider this unless the turbine lies in the same processor
if (FAST.get_procNo(iTurb) != NaluEnv::self().parallel_rank()) continue;

    // Identify if the turbine number has changed 
    if (iTurb != it) {
      it = iTurb;  // Initialize the turbine number
      ib = 0;  // Initialize the first blad
      actPtrCounter = -1;  // Initialize the first actuator point
    }

    // The total number of blades
    const int numBlades = static_cast<int>(FAST.get_numBlades(iTurb));
    // The total number of actuator points in all blades
    //~ const size_t totalActuatorNodes = FAST.get_numForcePtsBlade(iTurb);
    // The total number of actuator points per blade
    //~ const size_t ptsPerBlade = totalActuatorNodes / numBlades;
    const int ptsPerBlade = static_cast<int>(FAST.get_numForcePtsBlade(iTurb));

    // If the number of actuator points is greater than the number per blade
    //   then increase the blade number index 
    actPtrCounter++;
    if (actPtrCounter == ptsPerBlade) {
      ib++;
      actPtrCounter = 0;      
      }
    // Increment the actuator point counter
    //~ else {actPtrCounter++;}

    if (ib == numBlades) continue;

    // Store the actuator point into the counter index
    indexMap_[it][ib][actPtrCounter] = np;

  }

}  
////////////////////////////////////////////////////////////////////////////////


//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//-------- compute_elem_centroid -------------------------------------------
//--------------------------------------------------------------------------
void
ActuatorFAST::compute_elem_centroid(
  const int& nDim, double* elemCentroid, const int& nodesPerElement)
{
  // zero
  for (int j = 0; j < nDim; ++j)
    elemCentroid[j] = 0.0;

  // assemble
  for (int ni = 0; ni < nodesPerElement; ++ni) {
    for (int j = 0; j < nDim; ++j) {
      elemCentroid[j] += ws_coordinates_[ni * nDim + j] / nodesPerElement;
    }
  }
}

// Spread actuator force to nodes
void
ActuatorFAST::spread_actuator_force_to_node_vec(
  const int& nDim,
  std::set<stk::mesh::Entity>& nodeVec,
  const std::vector<double>& actuator_force,
  // The tensor to indicate the orientation of the airfoil sections
  const std::vector<double>& orientation_tensor,
  const double* actuator_node_coordinates,
  const stk::mesh::FieldBase& coordinates,
  stk::mesh::FieldBase& actuator_source,
  const stk::mesh::FieldBase& dual_nodal_volume,
  const Coordinates& epsilon,
  const std::vector<double>& hubPt,
  const std::vector<double>& hubShftDir,
  std::vector<double>& thr,
  std::vector<double>& tor)
{

  std::vector<double> ws_nodeForce(nDim);

  // This is the distance vector
  std::vector<double> distance(nDim);
  // This is the distance vector projected onto the blade coordinate system
  std::vector<double> distance_projected(nDim, 0.0);

  // Loop through all the mesh points influenced by the body force
  std::set<stk::mesh::Entity>::iterator iNode;
  for (iNode = nodeVec.begin(); iNode != nodeVec.end(); ++iNode) {

    stk::mesh::Entity node = *iNode;

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
///////////                Error is HERE
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
    const double* node_coords =
      (double*)stk::mesh::field_data(coordinates, node);

    const double* dVol =
      (double*)stk::mesh::field_data(dual_nodal_volume, node);

    // compute distance
    compute_distance(nDim, node_coords, actuator_node_coordinates,
                     distance.data());

    // Now project the distance into the blade reference frame
    // The ordering of the rotation matrix is: 
    //   xx [0], xy [1], xz [2], yx [3], yy [4], yz [5], zx [6], zy [7], zz [8]
    // First loop through every direction
    for (int j = 0; j < nDim; ++j){

      // Initialize the disnace to zero
      distance_projected[j] = 0.0;

      // Now compute the matrix multiplication
      // This implementation allows for 2D and 3D
      // This loop is used to go through elemnts of the rotation matrix
      // The projection from (x1, y1, z1) to (x2, y2, z2) is:
      // x2 = x1 * xx + y1 * yx + z1 * zx
      // y2 = x1 * xy + y1 * yy + z1 * zy
      // z2 = x1 * xz + y1 * yz + z1 * zz
      for (int k = 0; k < nDim; k++){

        // The coordinates in this distance have the first two element switched 
        // thickness, chord, spanwise (notice this is the OpenFAST definition)
        distance_projected[j] += distance[k] * orientation_tensor[j + k * nDim];
      }
    }
    // Switch components 0 and 1 to be consistent with OpenFAST
    // The new distances are given in:
    //   chord (0), thickness (1), and spanwise (2) directions
    distance_projected[0] = distance_projected[0] + distance_projected[1];
    distance_projected[1] = distance_projected[0] - distance_projected[1];
    distance_projected[0] = distance_projected[0] - distance_projected[1];
    
    // project the force to this node with projection function
    // To de-activate the projection use distance.data() instead of 
    //   distance_projected.data()
    double gA = actuator_utils::Gaussian_projection(nDim, distance.data(), epsilon);

    compute_node_force_given_weight(
      nDim, gA, &actuator_force[0], &ws_nodeForce[0]);

    double* sourceTerm = (double*)stk::mesh::field_data(actuator_source, node);
    for (int j = 0; j < nDim; ++j)
      sourceTerm[j] += ws_nodeForce[j];

    add_thrust_torque_contrib(
      nDim, node_coords, *dVol, ws_nodeForce, hubPt, hubShftDir, thr, tor);
  }
}

// Add to thrust and torque contribution from current node
void
ActuatorFAST::add_thrust_torque_contrib(
  const int& nDim,
  const double* nodeCoords,
  const double dVol,
  const std::vector<double>& nodeForce,
  const std::vector<double>& hubPt,
  const std::vector<double>& hubShftDir,
  std::vector<double>& thr,
  std::vector<double>& tor)
{

  std::vector<double> rPerpShft(nDim);
  std::vector<double> r(nDim);
  for (int j = 0; j < nDim; j++) {
    r[j] = nodeCoords[j] - hubPt[j];
  }
  double rDotHubShftVec =
    r[0] * hubShftDir[0] + r[1] * hubShftDir[1] + r[2] * hubShftDir[2];
  for (int j = 0; j < nDim; j++) {
    rPerpShft[j] = r[j] - rDotHubShftVec * hubShftDir[j];
  }

  for (int j = 0; j < nDim; j++) {
    thr[j] += nodeForce[j] * dVol;
  }
  tor[0] += (rPerpShft[1] * nodeForce[2] - rPerpShft[2] * nodeForce[1]) * dVol;
  tor[1] += (rPerpShft[2] * nodeForce[0] - rPerpShft[0] * nodeForce[2]) * dVol;
  tor[2] += (rPerpShft[0] * nodeForce[1] - rPerpShft[1] * nodeForce[0]) * dVol;
}

std::string
ActuatorFAST::write_turbine_points_to_string(
  std::size_t turbNum, int width, int precision)
{
  std::ostringstream stream;
  // header
  stream << std::setw(width) << std::left << "ID"
         << ",";
  stream << std::setw(width) << std::left << "TYPE"
         << ",";
  stream << std::setw(width) << std::left << "X"
         << ",";
  stream << std::setw(width) << std::left << "Y"
         << ",";
  stream << std::setw(width) << std::left << "Z";
  stream << std::endl;

  // data
  for (auto&& point : actuatorPointInfoMap_) {
    auto fastPoint = dynamic_cast<ActuatorFASTPointInfo*>(point.second.get());
    const std::size_t id = point.first;
    if (fastPoint->globTurbId_ == turbNum) {
      stream << std::setw(width) << std::setprecision(precision) << std::left
             << id << ",";
      stream << std::setw(width) << std::setprecision(precision) << std::left
             << fastPoint->nodeType_ << ",";
      stream << std::setw(width) << std::setprecision(precision) << std::left
             << fastPoint->centroidCoords_[0] << ",";
      stream << std::setw(width) << std::setprecision(precision) << std::left
             << fastPoint->centroidCoords_[1] << ",";
      stream << std::setw(width) << std::setprecision(precision) << std::left
             << fastPoint->centroidCoords_[2];
      stream << std::endl;
    }
  }

  return stream.str();
}

void
ActuatorFAST::dump_turbine_points_to_file(std::size_t turbNum)
{
  auto turbInfo =
    dynamic_cast<ActuatorFASTInfo*>(actuatorInfo_.at(turbNum).get());
  std::string fileToDumpTo = turbInfo->fileToDumpPoints_;

  if (
    !fileToDumpTo.empty() &&
    FAST.get_procNo(turbNum) == NaluEnv::self().parallel_rank()) {
    NaluEnv::self().naluOutput() << "Dumping turbine " << turbNum
                                 << " to file: " << fileToDumpTo << std::endl;
    std::ofstream csvOut;
    csvOut.open(fileToDumpTo, std::ofstream::out);
    std::string actOut = write_turbine_points_to_string(turbNum, 10, 8);
    csvOut << actOut;
    csvOut.close();
  }
}

} // namespace nalu
} // namespace sierra
