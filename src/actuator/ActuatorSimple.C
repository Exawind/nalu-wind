// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include <actuator/ActuatorSimple.h>
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
ActuatorSimpleInfo::ActuatorSimpleInfo() : ActuatorInfo(), fllt_correction_(false)
{}

// destructor
ActuatorSimpleInfo::~ActuatorSimpleInfo()
{
  // nothing to do
}

// constructor
ActuatorSimplePointInfo::ActuatorSimplePointInfo(
  size_t globTurbId,
  size_t bladeId,
  Point centroidCoords,
  double searchRadius,
  Coordinates epsilon,
  Coordinates epsilon_opt,
  int nType, //fast::ActuatorNodeType nType,
  int forceInd)
  : ActuatorPointInfo(
      centroidCoords, searchRadius, 1.0e16, stk::mesh::Entity()),
    globTurbId_(globTurbId),
    bladeId_(bladeId),
    epsilon_(epsilon),
    epsilon_opt_(epsilon_opt),
    nodeType_(nType),
    forcePntIndex_(forceInd)
{
  // nothing to do
}

// destructor
ActuatorSimplePointInfo::~ActuatorSimplePointInfo()
{
  // nothing to do
}

// constructor
ActuatorSimple::ActuatorSimple(Realm& realm, const YAML::Node& node)
  : Actuator(realm, node), numFastPoints_(0)
{
  // Actuator::load(node);
  // load the data
  load(node);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ActuatorSimple::~ActuatorSimple()
{
}

// Multiply the point force by the weight at this element location.
void
ActuatorSimple::compute_node_force_given_weight(
  const int& nDim, const double& g, const double* pointForce, double* nodeForce)
{

  for (int j = 0; j < nDim; ++j)
    nodeForce[j] = pointForce[j] * g;
}

//--------------------------------------------------------------------------
//-------- load ------------------------------------------------------------
//--------------------------------------------------------------------------
void
ActuatorSimple::load(const YAML::Node& y_node)
{
  // check for any data probes
  const YAML::Node y_actuator = y_node["actuator"];
  if (y_actuator) {

    // --- Stuff to load the simple blade ---
    const YAML::Node debug_output = y_actuator["debug_output"];
    if (debug_output) 
      debug_output_ = debug_output.as<bool>();
    else
      debug_output_ = false;
    
    get_required(y_actuator, "n_simpleblades", n_simpleblades_);
    if (n_simpleblades_ > 0) {
      for (int iBlade= 0; iBlade < n_simpleblades_; iBlade++) {
	NaluEnv::self().naluOutputP0() << "Reading blade: " << iBlade<< std::endl; //LCCOUT
	const YAML::Node cur_blade =
	  y_actuator["Blade" + std::to_string(iBlade)];

	actuatorInfo_.emplace_back(new ActuatorSimpleInfo());
	auto actuatorSimpleInfo =
	  dynamic_cast<ActuatorSimpleInfo*>(actuatorInfo_.back().get());

	actuatorSimpleInfo->isSimpleBlade_ = true;
	actuatorSimpleInfo->runOnProc_     = 0; // FIX THIS LATER
	actuatorSimpleInfo->bladeId_       = iBlade;

	const YAML::Node numbladepts = cur_blade["num_force_pts_blade"];
	if (numbladepts) 
	  actuatorSimpleInfo->num_force_pts_blade_ = numbladepts.as<size_t>() ;
	else
	  throw std::runtime_error("ActuatorSimple: missing num_force_pts_blade");
	if (debug_output_)
	  NaluEnv::self().naluOutputP0() 
	    << "Reading blade: " << iBlade
	    << " num_force_pts_blade: "
	    << actuatorSimpleInfo->num_force_pts_blade_ << std::endl; //LCCOUT

	std::string bladeFileName;
	get_if_present(cur_blade, "file_to_dump_turb_pts", bladeFileName);
	if (!bladeFileName.empty()) {
	  actuatorSimpleInfo->fileToDumpPoints_ = bladeFileName;
	}

          // The value epsilon / chord [non-dimensional]
          // This is a vector containing the values for:
          //   - chord aligned (x),
          //   - tangential to chord (y),
          //   - spanwise (z)
          const YAML::Node epsilon_chord = cur_blade["epsilon_chord"];
          const YAML::Node epsilon = cur_blade["epsilon"];
          if(epsilon && epsilon_chord){
            throw std::runtime_error("epsilon and epsilon_chord have both been specified for Turbine "
              + std::to_string(iBlade) + "\nYou must pick one or the other.");
          }
          if(epsilon && actuatorSimpleInfo->fllt_correction_){
            throw std::runtime_error("epsilon and fllt_correction have both been specified for Turbine "
              +std::to_string(iBlade) + "\nepsilon_chord and epsilon_min should be used with fllt_correction.");
          }

	  // ** handle epsilon stuff **
          // If epsilon/chord is given, store it,
          // If it is not given, set it to zero, such
          // that it is smaller than the standard epsilon and
          // will not be used
          if ( epsilon_chord )
          {
            // epsilon / chord
            actuatorSimpleInfo->epsilon_chord_ = epsilon_chord.as<Coordinates>();

            // Minimum epsilon allowed in simulation. This is required when
            //   specifying epsilon/chord
            get_required(cur_blade, "epsilon_min",
              actuatorSimpleInfo->epsilon_min_);
          }
          // Set all unused epsilon values to zero
          else
          {
            actuatorSimpleInfo->epsilon_chord_.x_ = 0.;
            actuatorSimpleInfo->epsilon_chord_.y_ = 0.;
            actuatorSimpleInfo->epsilon_chord_.z_ = 0.;
            actuatorSimpleInfo->epsilon_min_.x_ = 0.;
            actuatorSimpleInfo->epsilon_min_.y_ = 0.;
            actuatorSimpleInfo->epsilon_min_.z_ = 0.;
          }

          // Check if epsilon is given and store it.
          if ( epsilon ) {
            // Store the minimum epsilon
            actuatorSimpleInfo->epsilon_ = epsilon.as<Coordinates>();
          }
          // If epsilon/chord is given and not standard epsilon, then assign
          //   the minimum epsilon as standard epsilon
          else  if (epsilon_chord) {
            // Get the minimum epsilon
            actuatorSimpleInfo->epsilon_ = actuatorSimpleInfo->epsilon_min_;
          }
          // If none of the conditions are met, throw an error
          else {
            throw std::runtime_error(
              "ActuatorLineSimple: lacking epsilon vector");
          }

	  // Handle blade properties
          const YAML::Node p1 = cur_blade["p1"];
	  if (p1) 
	    actuatorSimpleInfo->p1_ = p1.as<sierra::nalu::Coordinates>() ;
	  else
	    throw std::runtime_error("ActuatorSimple: missing p1");
          const YAML::Node p2 = cur_blade["p2"];
	  if (p2) 
	    actuatorSimpleInfo->p2_ = p2.as<sierra::nalu::Coordinates>() ;
	  else
	    throw std::runtime_error("ActuatorSimple: missing p2");
          const YAML::Node p1zeroAOAnode = cur_blade["p1_zero_alpha_dir"];
	  Coordinates p1zeroAOA;
	  if (p1zeroAOAnode) {
	    // Normalize and save p1zeroAOA
	    p1zeroAOA = p1zeroAOAnode.as<sierra::nalu::Coordinates>();
	    double norm = sqrt(p1zeroAOA.x_*p1zeroAOA.x_ + p1zeroAOA.y_*p1zeroAOA.y_ + p1zeroAOA.z_*p1zeroAOA.z_);
	    p1zeroAOA.x_ = p1zeroAOA.x_/norm;
	    p1zeroAOA.y_ = p1zeroAOA.y_/norm;
	    p1zeroAOA.z_ = p1zeroAOA.z_/norm;
	    actuatorSimpleInfo->p1zeroalphadir_ = p1zeroAOA;
	  } else
	    throw std::runtime_error("ActuatorSimple: missing p1_zero_alpha_dir");	    
	  // Compute span and chord normal direction
	  Coordinates spandir;
	  spandir.x_ = actuatorSimpleInfo->p2_.x_ - actuatorSimpleInfo->p1_.x_;
	  spandir.y_ = actuatorSimpleInfo->p2_.y_ - actuatorSimpleInfo->p1_.y_;
	  spandir.z_ = actuatorSimpleInfo->p2_.z_ - actuatorSimpleInfo->p1_.z_;
	  double norm = sqrt(spandir.x_*spandir.x_ + spandir.y_*spandir.y_ +
			     spandir.z_*spandir.z_);
	  spandir.x_ = spandir.x_/norm;
	  spandir.y_ = spandir.y_/norm;
	  spandir.z_ = spandir.z_/norm;
	  actuatorSimpleInfo->spandir_ = spandir;

	  Coordinates chordnormaldir;
	  chordnormaldir.x_ = p1zeroAOA.y_*spandir.z_ - p1zeroAOA.z_*spandir.y_;
	  chordnormaldir.y_ = p1zeroAOA.z_*spandir.x_ - p1zeroAOA.x_*spandir.z_;
	  chordnormaldir.z_ = p1zeroAOA.x_*spandir.y_ - p1zeroAOA.y_*spandir.x_;
	  actuatorSimpleInfo->chordnormaldir_ = chordnormaldir;

	  // output directions
	  if (debug_output_) {
	    NaluEnv::self().naluOutputP0()  // LCCOUT
	      << "Blade: " << iBlade << " p1zeroAOA dir: "
	      <<p1zeroAOA.x_<<" "<<p1zeroAOA.y_<<" "<<p1zeroAOA.z_<< std::endl;
	    NaluEnv::self().naluOutputP0()  // LCCOUT
	      << "Blade: " << iBlade << " Span dir: "
	      <<spandir.x_<<" "<<spandir.y_<<" "<<spandir.z_<< std::endl; 
	    NaluEnv::self().naluOutputP0() // LCCOUT
	      << "Blade: " << iBlade 
	      << " chord norm dir: "<<std::setprecision(5)
	      <<chordnormaldir.x_<<" "<<chordnormaldir.y_<<" "<<chordnormaldir.z_<< std::endl; 
	  }

	  // Chord definitions
          const YAML::Node chord_table = cur_blade["chord_table"];
	  std::vector<double> chordtemp;
	  if (chord_table) 
	    chordtemp = chord_table.as<std::vector<double>>();
	  else 
	    throw std::runtime_error("ActuatorSimple: missing chord_table");

	  // twist definitions
          const YAML::Node twist_table = cur_blade["twist_table"];
	  std::vector<double> twisttemp;
	  if (twist_table)
	    twisttemp = twist_table.as<std::vector<double>>();
	  else
	    throw std::runtime_error("ActuatorSimple: missing twist_table");

	  // Sanitize chord and twist tables
	  actuatorSimpleInfo->chord_table_ = 
	    extend_double_vector(chordtemp, actuatorSimpleInfo->num_force_pts_blade_);
	  actuatorSimpleInfo->twist_table_ = 
	    extend_double_vector(twisttemp, actuatorSimpleInfo->num_force_pts_blade_);
	  

	  // Get the element areas
	  actuatorSimpleInfo->elem_area_ = 
	    get_blade_area_elems(actuatorSimpleInfo->chord_table_ ,
				 actuatorSimpleInfo->p1_,
				 actuatorSimpleInfo->p2_,
				 actuatorSimpleInfo->num_force_pts_blade_);

	  // Handle polar tables
	  // Get AOA
          const YAML::Node aoa_table = cur_blade["aoa_table"];
	  if (aoa_table)
	    actuatorSimpleInfo->aoa_polartable_ = aoa_table.as<std::vector<double>>();
	  else
	    throw std::runtime_error("ActuatorSimple: missing aoa_table");
	  size_t polartableN = actuatorSimpleInfo->aoa_polartable_.size();
	  // Get CL
          const YAML::Node cl_table = cur_blade["cl_table"];
	  if (cl_table) {
	    std::vector<double> cltablevec = cl_table.as<std::vector<double>>();
	    actuatorSimpleInfo->cl_polartable_ = 
	      extend_double_vector(cltablevec, polartableN);
	  }
	  else
	    throw std::runtime_error("ActuatorSimple: missing cl_table");
	  // Get CD
          const YAML::Node cd_table = cur_blade["cd_table"];
	  if (cd_table) {
	    std::vector<double> cdtablevec = cd_table.as<std::vector<double>>();
	    actuatorSimpleInfo->cd_polartable_ = 
	      extend_double_vector(cdtablevec, polartableN);
	  }
	  else
	    throw std::runtime_error("ActuatorSimple: missing cd_table");
	  NaluEnv::self().naluOutputP0() << "ActuatorSimple::loaded blade "<<iBlade << std::endl;

      }
      //LCCSTOP throw std::runtime_error("ActuatorSimple: done loading blades");
    } else {
      throw std::runtime_error("Number of simple blades <= 0 ");
    }


  }
}

/** Called after load, but before initialize. The mesh isn't loaded yet. For
   now, this function only checks that the Nalu time step is an integral
   multiple of the FAST time step
*/
void
ActuatorSimple::setup()
{
  // objective: declare the part, register coordinates; must be before
  // populate_mesh()

}

/** This function searches for the processor containing the hub point of each
 * turbine and allocates the turbine to that processor. It does this through a
 * stk::coarse_search of bounding boxes around the processor domains.
 */
void
ActuatorSimple::allocateTurbinesToProcs()
{
  stk::mesh::MetaData& metaData = realm_.meta_data();
  const int nDim = metaData.spatial_dimension();

  // initialize thrust and torque
  thrust.resize(n_simpleblades_);
  torque.resize(n_simpleblades_);
  for (size_t iBlade = 0; iBlade < n_simpleblades_; ++iBlade) {
    thrust[iBlade].resize(nDim);
    torque[iBlade].resize(nDim);
  }

  // initialize BladeTotalLift and BladeTotalDrag
  BladeTotalLift.resize(n_simpleblades_);
  BladeTotalDrag.resize(n_simpleblades_);
  BladeAvgAlpha.resize(n_simpleblades_);
  BladeAvgWS2D.resize(n_simpleblades_);
  for (size_t iBlade = 0; iBlade < n_simpleblades_; ++iBlade) {
    BladeAvgWS2D[iBlade].resize(nDim);
  }
}

/** This method allocates the turbines to processors, initializes the OpenFAST
 * instances of each turbine, populates the map of ActuatorLinePointInfo with
 * the actuator points of all turbines and determines the elements associated
 * with each actuator point.
 */
void
ActuatorSimple::initialize()
{

  allocateTurbinesToProcs();

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

  //throw std::runtime_error("ActuatorSimple: create_actuator_point_info_map()");  //LCCSTOP

  create_point_info_map_class_specific();

  // coarse search
  determine_elems_to_ghost();

  // manage ghosting
  manage_ghosting();

  // complete filling in the set of elements connected to the centroid
  complete_search();

  //throw std::runtime_error("ActuatorSimple: done initialize()");  //LCCSTOP

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
ActuatorSimple::update()
{

}

/** This function is called at each time step. This samples the velocity at each
 * actuator point, advances the OpenFAST turbine models to Nalu's next time step
 * and assembles the source terms in the momentum equation for Nalu.
 */
void
ActuatorSimple::execute()
{
  //LCC throw std::runtime_error("ActuatorSimple: start execute()");  //LCCSTOP

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
      dynamic_cast<ActuatorSimplePointInfo*>(actuatorPointInfoMap_.at(np).get());
    if (infoObject == NULL) {
      throw std::runtime_error("Object in ActuatorPointInfo is not the correct "
                               "type.  Should be ActuatorSimplePointInfo.");
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

    // Set properties at the point
    infoObject->gasDensity_ = ws_pointGasDensity;
    infoObject->windSpeed_.x_ = ws_pointGasVelocity[0];
    infoObject->windSpeed_.y_ = ws_pointGasVelocity[1];
    if (nDim>2) infoObject->windSpeed_.z_ = ws_pointGasVelocity[2];

  }

  // Reset thrust and torque
  for (size_t iBlade = 0; iBlade < n_simpleblades_; ++iBlade) {
    BladeTotalLift[iBlade] = 0.0;
    BladeTotalDrag[iBlade] = 0.0;
    BladeAvgAlpha[iBlade]  = 0.0;
    for (int j = 0; j < nDim; j++) {
      torque[iBlade][j] = 0.0;
      thrust[iBlade][j] = 0.0;
      BladeAvgWS2D[iBlade][j] = 0.0;
    }
  }

  // if disk average azimuthally
  // apply the forcing terms
  execute_class_specific(nDim, coordinates, actuator_source, dualNodalVolume);

  // Write the thrust outputs
  NaluEnv::self().naluOutputP0() << " -- START BLADE summary --" <<std::endl;
  // Outputs from source terms
  for (size_t iBlade =0; iBlade < n_simpleblades_; iBlade++) {
    auto bladeInfo =
      dynamic_cast<ActuatorSimpleInfo*>(actuatorInfo_.at(iBlade).get());

    if (NaluEnv::self().parallel_rank() == bladeInfo->runOnProc_) 
      NaluEnv::self().naluOutput()
	<< " ALS Blade "<<iBlade<<" Force: "<<std::setprecision(8)
	<< thrust[iBlade][0] << " "
	<< thrust[iBlade][1] << " " << thrust[iBlade][2] << " " << std::endl;
  }
  // Outputs from BEM theory
  for (size_t iBlade =0; iBlade < n_simpleblades_; iBlade++) {
    auto bladeInfo =
      dynamic_cast<ActuatorSimpleInfo*>(actuatorInfo_.at(iBlade).get());
    
    if (NaluEnv::self().parallel_rank() == bladeInfo->runOnProc_) 
      {
	NaluEnv::self().naluOutput()
	  << " BEM Blade "<<iBlade<<" Lift: "<<std::setprecision(8)
	  << BladeTotalLift[iBlade]<<" Drag: "
	  << BladeTotalDrag[iBlade]<<std::endl;
      }
  }
  // AVG outputs
  for (size_t iBlade =0; iBlade < n_simpleblades_; iBlade++) {
    auto bladeInfo =
      dynamic_cast<ActuatorSimpleInfo*>(actuatorInfo_.at(iBlade).get());
    const size_t Npts = bladeInfo->num_force_pts_blade_;
    if (NaluEnv::self().parallel_rank() == bladeInfo->runOnProc_) 
      {
	NaluEnv::self().naluOutput()
	  << " AVG Blade "<<iBlade<<" Alpha: "<<std::setprecision(8)
	  << BladeAvgAlpha[iBlade]/(float)Npts << " WS: "
	  << BladeAvgWS2D[iBlade][0]/(float)Npts << " "
	  << BladeAvgWS2D[iBlade][1]/(float)Npts << " "
	  << BladeAvgWS2D[iBlade][2]/(float)Npts << " "
	  << std::endl;
      }
  }
  NaluEnv::self().naluOutputP0() << " -- END summary --" <<std::endl;


  //throw std::runtime_error("ActuatorSimple: done execute_class_specific()");  //LCCSTOP

  // parallel assemble (contributions from ghosted and locally owned)
  const std::vector<const stk::mesh::FieldBase*> sumFieldVec(
    1, actuator_source);
  stk::mesh::parallel_sum_including_ghosts(bulkData, sumFieldVec);

  const std::vector<const stk::mesh::FieldBase*> sumFieldG(1, g);
  stk::mesh::parallel_sum_including_ghosts(bulkData, sumFieldG);
}


// Copmute the filtered lifting line theory correction
void ActuatorSimple::filtered_lifting_line()
{

}

// Creates bounding boxes around the subdomain of each processor
void
ActuatorSimple::populate_candidate_procs()
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
        "ActuatorSimple: Part is null" + searchTargetNames_[k]);
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
ActuatorSimple::create_actuator_point_info_map()
{

  stk::mesh::MetaData& metaData = realm_.meta_data();
  const int nDim = metaData.spatial_dimension();

  size_t np = 0;

  // Do simple blade stuff
  for (size_t iBlade = 0; iBlade < actuatorInfo_.size(); ++iBlade) {
    if (debug_output_)
      NaluEnv::self().naluOutputP0()
	<< "create_actuator_point_info_map: " << iBlade<< std::endl; //LCCOUT 

    const auto actuatorInfo =
      dynamic_cast<ActuatorSimpleInfo*>(actuatorInfo_[iBlade].get());
    if (actuatorInfo == NULL) {
      throw std::runtime_error("Object in actuatorInfo is not the correct "
                               "type.  It should be ActuatorSimpleInfo.");
    }
    if (!actuatorInfo->isSimpleBlade_) {
      throw std::runtime_error("Object in actuatorInfo is not Simple Blade");
    }
    
    int processorId = actuatorInfo->runOnProc_;
    if (processorId == NaluEnv::self().parallel_rank()) {

      // define a point that will hold the centroid
      Point centroidCoords;

      // scratch array for coordinates and dummy array for velocity
      std::vector<double> currentCoords(3, 0.0);

      // loop over all points for this turbine
      const int numForcePts = actuatorInfo->num_force_pts_blade_;
      for (int iNode = 0; iNode < numForcePts; iNode++) {
	stk::search::IdentProc<uint64_t, int> theIdent(
            np, NaluEnv::self().parallel_rank());
	
	// set model coordinates from FAST
	// move the coordinates; set the velocity... may be better on the
	// lineInfo object
	//getBladeCoordinates(currentCoords, iNode);
	get_blade_coordinates(nDim, currentCoords, 
			      actuatorInfo->p1_, actuatorInfo->p2_,
			      actuatorInfo->num_force_pts_blade_, iNode);

	// Get the chord from inside of FAST to compute epsilon
	//double chord = FAST.getChord(np, iTurb);
	double chord = get_blade_chord(actuatorInfo->chord_table_, iNode);

	// create the point info and push back to map
	Coordinates epsilon;
	// This is the optimal epsilon
	Coordinates epsilon_opt;

	// Assume all points are BLADE type
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
	int NODETYPE=0;
	actuatorPointInfoMap_.insert(std::make_pair(
                     np, make_unique<ActuatorSimplePointInfo>(
                           iBlade, 
			   iBlade,
                           centroidCoords, 
                           searchRadius, 
                           epsilon,
                           epsilon_opt,
                           NODETYPE, iNode)));

	// Print the value of epsilon to the screen
	if (debug_output_)
	  NaluEnv::self().naluOutput()
	    << "  Actuator point "
	    << iNode
	    << " : "
	    << centroidCoords[0] << " "
	    << centroidCoords[1] << " "
	    << centroidCoords[2] << " eps: "	  
	    << epsilon.x_ << " "
	    << epsilon.y_ << " "
	    << epsilon.z_ << " "
	    << " " << std::endl;
	
	// Counter for the number of blade points
	np = np + 1;
      } // Loop over iNode

    }

  }
  numFastPoints_ = actuatorPointInfoMap_.size();

  // execute this outside of loop so actuatorPoints that go into fast are
  // contiguous in the vectors (i.e. the first FAST.get_numForcePts() number of
  // points are the actuator lines, and whatever comes after them are the swept
  // points)
  create_point_info_map_class_specific();

  //index_map();
  
}

//--------------------------------------------------------------------------
//-------- update_actuator_line_point_info_map -----------------------------
//--------------------------------------------------------------------------
void
ActuatorSimple::update_actuator_point_info_map()
{

  stk::mesh::MetaData& metaData = realm_.meta_data();
  const int nDim = metaData.spatial_dimension();

  size_t np = 0;

}

/// This function computes the index map such that actuator points can be
///   accessed using indexing:
///   (turbine number, blade number, actuator point number)
void ActuatorSimple::index_map()
{

}
////////////////////////////////////////////////////////////////////////////////

// Get the (x,y,z) coordinates of that blade at node iNode
void 
ActuatorSimple::get_blade_coordinates(
  const int& nDim, std::vector<double> &coord, 
  const Coordinates &p1,  const Coordinates &p2, 
  const int &Npts, const int &iNode)
{
  std::vector<double> dx(nDim, 0.0);
  double denom = (double)Npts; //Npts - 1.0;
  dx[0] = (p2.x_ - p1.x_)/denom; 
  dx[1] = (p2.y_ - p1.y_)/denom; 
  if (nDim>2) dx[2] = (p2.z_ - p1.z_)/denom; 
  
  coord[0] = p1.x_ + 0.5*dx[0] +  dx[0]*(float)iNode;
  coord[1] = p1.y_ + 0.5*dx[1] + dx[1]*(float)iNode;
  if (nDim>2) coord[2] = p1.z_ + 0.5*dx[2] + dx[2]*(float)iNode;

}

// Get the chord length at that location
double 
ActuatorSimple::get_blade_chord(
        std::vector<double> &chord_table,
	const int& iNode)
{
  return chord_table[iNode];
}

std::vector<double> 
ActuatorSimple::extend_double_vector(std::vector<double> vec, const int N)
{
  if ((vec.size() != 1) && (vec.size() != N))
    throw std::runtime_error("Vector is not of size 1 or "+std::to_string(N));
  if (vec.size() == 1) 
    { // Extend the vector to size N
      std::vector<double> newvec(N, vec[0]);
      return newvec;
    }
  if (vec.size() == N) 
    return vec;
  return vec;  // Should not get here
}

std::vector<double> 
ActuatorSimple::get_blade_area_elems(
   std::vector<double> chord_table, 
   const Coordinates &p1,  
   const Coordinates &p2,
   const int &Npts)
{
  // stk::mesh::MetaData& metaData = realm_.meta_data();
  // const int nDim = metaData.spatial_dimension();
  const int nDim = 3;
  std::vector<double> areas(Npts, 0.0);
  double denom = (double)Npts; //Npts - 1.0;
  std::vector<double> dx(nDim, 0.0);

  dx[0] = (p2.x_ - p1.x_)/denom; 
  dx[1] = (p2.y_ - p1.y_)/denom; 
  if (nDim>2) dx[2] = (p2.z_ - p1.z_)/denom; 
  // Assumes equal area spacing
  double dx_norm = sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
  for (int i=0; i<chord_table.size(); i++) {
    areas[i] = dx_norm*chord_table[i];
  }
  
  return areas;
}


//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//-------- compute_elem_centroid -------------------------------------------
//--------------------------------------------------------------------------
void
ActuatorSimple::compute_elem_centroid(
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
ActuatorSimple::spread_actuator_force_to_node_vec(
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
ActuatorSimple::add_thrust_torque_contrib(
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
ActuatorSimple::write_turbine_points_to_string(
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
    auto fastPoint = dynamic_cast<ActuatorSimplePointInfo*>(point.second.get());
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
ActuatorSimple::dump_turbine_points_to_file(std::size_t turbNum)
{
  auto turbInfo =
    dynamic_cast<ActuatorSimpleInfo*>(actuatorInfo_.at(turbNum).get());
  std::string fileToDumpTo = turbInfo->fileToDumpPoints_;
}

} // namespace nalu
} // namespace sierra
