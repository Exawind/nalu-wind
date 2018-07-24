/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifdef NALU_USES_OPENFAST

#include <ActuatorLineFAST.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>
#include <NaluEnv.h>
#include <Realm.h>
#include <Simulation.h>

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

namespace sierra{
namespace nalu{


// constructor
ActuatorLineFASTInfo::ActuatorLineFASTInfo()
  : processorId_(0),
    numPoints_(1),
    turbineName_("machine_one")
{
  // nothing to do
}


// destructor
ActuatorLineFASTInfo::~ActuatorLineFASTInfo()
{
  // nothing to do
}


// constructor
ActuatorLineFASTPointInfo::ActuatorLineFASTPointInfo(
  size_t globTurbId,
  Point centroidCoords,
  double searchRadius,
  Coordinates epsilon,
  fast::ActuatorNodeType nType
  )
  : globTurbId_(globTurbId),
    centroidCoords_(centroidCoords),
    searchRadius_(searchRadius),
    epsilon_(epsilon),
    bestX_(1.0e16),
    bestElem_(stk::mesh::Entity()),
    nodeType_(nType)
{
  // nothing to do
}


// destructor
ActuatorLineFASTPointInfo::~ActuatorLineFASTPointInfo()
{
  // nothing to do
}


// constructor
ActuatorLineFAST::ActuatorLineFAST(
  Realm &realm,
  const YAML::Node &node)
  : Actuator(realm, node),
    realm_(realm),
    searchMethod_(stk::search::KDTREE),
    actuatorLineGhosting_(NULL),
    needToGhostCount_(0)
{
  // load the data
  load(node);

}


//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ActuatorLineFAST::~ActuatorLineFAST()
{

  FAST.end(); // Call destructors in FAST_cInterface

  // delete data probes specifications vector
  for ( size_t k = 0; k < actuatorLineInfo_.size(); ++k )
    delete actuatorLineInfo_[k];

  // clear actuatorLinePointInfoMap_
  for(auto iterPoint : actuatorLinePointInfoMap_)
  {
    delete iterPoint.second;
  }
}


// Multiply the point force by the weight at this element location.
void
ActuatorLineFAST::compute_node_force_given_weight(
  const int &nDim,
  const double &g,
  const double *pointForce,
  double *nodeForce)
{

  for ( int j = 0; j < nDim; ++j )
    nodeForce[j] = pointForce[j]*g;
}

/**
 * This method calculates the isotropic Gaussian projection of width epsilon of
 * a unit body force at the actuator point to another point at a distance *dis*
 * \f[
 * g(dis) = \frac{1}{\pi^{3/2}} \epsilon^3} e^{-\left( dis/ \epsilon \right)^2}
 * \f]
*/
double
ActuatorLineFAST::isotropic_Gaussian_projection(
  const int &nDim,
  const double &dis,
  const Coordinates &epsilon)
{
  // Compute the force projection weight at this location using an
  // isotropic Gaussian.
  double g;
  const double pi = acos(-1.0);
  if ( nDim == 2 )
    g = (1.0 / (pow(epsilon.x_,2.0) * pi)) * exp(-pow((dis/epsilon.x_),2.0));
  else
    g = (1.0 / (pow(epsilon.x_,3.0) * pow(pi,1.5))) * exp(-pow((dis/epsilon.x_),2.0));

  return g;
}


//--------------------------------------------------------------------------
//-------- load ------------------------------------------------------------
//--------------------------------------------------------------------------
void
ActuatorLineFAST::load(
  const YAML::Node & y_node)
{
  // check for any data probes
  const YAML::Node y_actuatorLine = y_node["actuator"];
  if (y_actuatorLine) {
    NaluEnv::self().naluOutputP0() << "ActuatorLineFAST::load" << std::endl;

    // search specifications
    std::string searchMethodName = "na";
    get_if_present(y_actuatorLine, "search_method", searchMethodName, searchMethodName);

    // determine search method for this pair
    if ( searchMethodName == "boost_rtree" )
      searchMethod_ = stk::search::BOOST_RTREE;
    else if ( searchMethodName == "stk_kdtree" )
      searchMethod_ = stk::search::KDTREE;
    else
      NaluEnv::self().naluOutputP0() << "ActuatorLineFAST::search method not declared; will use stk_kdtree" << std::endl;

    // extract the set of from target names; each spec is homogeneous in this respect
    const YAML::Node searchTargets = y_actuatorLine["search_target_part"];
    if (searchTargets.Type() == YAML::NodeType::Scalar) {
      searchTargetNames_.resize(1);
      searchTargetNames_[0] = searchTargets.as<std::string>() ;
    }
    else {
      searchTargetNames_.resize(searchTargets.size());
      for (size_t i=0; i < searchTargets.size(); ++i) {
        searchTargetNames_[i] = searchTargets[i].as<std::string>() ;
      }
    }

    // Populate object of inputs class to FAST
    fi.comm = NaluEnv::self().parallel_comm() ;

    get_required(y_actuatorLine, "n_turbines_glob", fi.nTurbinesGlob);

    if (fi.nTurbinesGlob > 0) {

      get_if_present(y_actuatorLine, "dry_run", fi.dryRun, false);
      get_if_present(y_actuatorLine, "debug", fi.debug, false);
      get_required(y_actuatorLine, "t_start", fi.tStart);
      std::string simStartType = "na";
      get_required(y_actuatorLine, "simStart", simStartType);
      if (simStartType == "init") {
	if (fi.tStart == 0) {
	  fi.simStart = fast::init;
	}
	else {
	  throw std::runtime_error("actuator_line: simStart type not consistent with start time for FAST");
	}
      } else if(simStartType == "trueRestart") {
	fi.simStart = fast::trueRestart;
      } else if(simStartType == "restartDriverInitFAST") {
	fi.simStart = fast::restartDriverInitFAST;
      }
      get_required(y_actuatorLine, "n_every_checkpoint", fi.nEveryCheckPoint);
      get_required(y_actuatorLine, "dt_fast", fi.dtFAST);
      get_required(y_actuatorLine, "t_max", fi.tMax); // tMax is the total duration to which you want to run FAST. This should be the same or greater than the max time given in the FAST fst file. Choose this carefully as FAST writes the output file only at this point if you choose the binary file output.

      if(y_actuatorLine["super_controller"]) {
          get_required(y_actuatorLine, "super_controller", fi.scStatus);
          get_required(y_actuatorLine, "sc_libFile", fi.scLibFile);
          get_required(y_actuatorLine, "num_sc_inputs", fi.numScInputs);
          get_required(y_actuatorLine, "num_sc_outputs", fi.numScOutputs);
      }

      fi.globTurbineData.resize(fi.nTurbinesGlob);
      for (int iTurb=0; iTurb < fi.nTurbinesGlob; iTurb++) {
	if (y_actuatorLine["Turbine" + std::to_string(iTurb)]) {

	  const YAML::Node cur_turbine = y_actuatorLine["Turbine"+std::to_string(iTurb)];
	  ActuatorLineFASTInfo *actuatorLineInfo = new ActuatorLineFASTInfo();
	  actuatorLineInfo_.push_back(actuatorLineInfo);

          get_required(cur_turbine, "turbine_name", actuatorLineInfo->turbineName_)  ;

	  // Force projection function properties
	  const YAML::Node epsilon = cur_turbine["epsilon"];
	  if ( epsilon )
	    actuatorLineInfo->epsilon_ = epsilon.as<Coordinates>() ;
	  else
	    throw std::runtime_error("ActuatorLineFAST: lacking epsilon vector");

	  readTurbineData(iTurb, fi, y_actuatorLine["Turbine" + std::to_string(iTurb)] );
	} else {
	  throw std::runtime_error("Node for Turbine" + std::to_string(iTurb) + " not present in input file or I cannot read it");
	}
      }

    } else {
      throw std::runtime_error("Number of turbines <= 0 ");
    }

    FAST.setInputs(fi);
  }
}

void ActuatorLineFAST::readTurbineData(int iTurb, fast::fastInputs & fi, YAML::Node turbNode) {

    //Read turbine data for a given turbine using the YAML node
    get_required(turbNode, "turb_id", fi.globTurbineData[iTurb].TurbID);
    get_required(turbNode, "fast_input_filename", fi.globTurbineData[iTurb].FASTInputFileName);
    get_required(turbNode, "restart_filename", fi.globTurbineData[iTurb].FASTRestartFileName);
    if (turbNode["turbine_base_pos"].IsSequence() ) {
        fi.globTurbineData[iTurb].TurbineBasePos = turbNode["turbine_base_pos"].as<std::vector<double> >() ;
    }
    if (turbNode["turbine_hub_pos"].IsSequence() ) {
        fi.globTurbineData[iTurb].TurbineHubPos = turbNode["turbine_hub_pos"].as<std::vector<double> >() ;
    }
    get_required(turbNode, "num_force_pts_blade", fi.globTurbineData[iTurb].numForcePtsBlade);
    get_required(turbNode, "num_force_pts_tower", fi.globTurbineData[iTurb].numForcePtsTwr);
    fi.globTurbineData[iTurb].nacelle_cd = 0.0;
    fi.globTurbineData[iTurb].nacelle_area = 0.0;
    fi.globTurbineData[iTurb].air_density = 0.0;
    get_if_present(turbNode, "nacelle_cd", fi.globTurbineData[iTurb].nacelle_cd);
    get_if_present(turbNode, "nacelle_area", fi.globTurbineData[iTurb].nacelle_area);
    get_if_present(turbNode, "air_density", fi.globTurbineData[iTurb].air_density);

}


/** Called after load, but before initialize. The mesh isn't loaded yet. For now, this function only
    checks that the Nalu time step is an integral multiple of the FAST time step
*/
void
ActuatorLineFAST::setup()
{
  // objective: declare the part, register coordinates; must be before populate_mesh()

  double dtNalu = realm_.get_time_step_from_file();
  tStepRatio_ = dtNalu/fi.dtFAST ;
  if (std::abs(dtNalu - tStepRatio_ * fi.dtFAST) < 0.001) {// TODO: Fix arbitrary number 0.001
    NaluEnv::self().naluOutputP0() << "Time step ratio  dtNalu/dtFAST: " << tStepRatio_ << std::endl ;
  } else {
    throw std::runtime_error("ActuatorLineFAST: Ratio of Nalu's time step is not an integral multiple of FAST time step");
  }

}

/** This function searches for the processor containing the hub point of each turbine and allocates the turbine
 *  to that processor. It does this through a stk::coarse_search of bounding boxes around the processor domains.
*/
void
ActuatorLineFAST::allocateTurbinesToProcs()
{
  stk::mesh::MetaData & metaData = realm_.meta_data();

  // clear some of the search info
  boundingHubSphereVec_.clear();
  boundingProcBoxVec_.clear();
  searchKeyPair_.clear();

  const int nDim = metaData.spatial_dimension();

  // set all of the candidate elements in the search target names
  populate_candidate_procs();

  const size_t nTurbinesGlob = FAST.get_nTurbinesGlob() ;
  thrust.resize(nTurbinesGlob);
  torque.resize(nTurbinesGlob);
  for (size_t iTurb = 0; iTurb < nTurbinesGlob; ++iTurb) {
    thrust[iTurb].resize(nDim);
    torque[iTurb].resize(nDim);
  }

  for (size_t iTurb = 0; iTurb < nTurbinesGlob; ++iTurb) {

    theKey theIdent(NaluEnv::self().parallel_rank(), NaluEnv::self().parallel_rank());

    // define a point that will hold the hub location
    Point hubPointCoords;
    std::vector<double> hubCoords (3, 0.0);
    FAST.getApproxHubPos(hubCoords, iTurb);
    for ( int j = 0; j < nDim; ++j )  hubPointCoords[j] = hubCoords[j];
    boundingSphere theSphere( Sphere(hubPointCoords, 1.0), theIdent);
    boundingHubSphereVec_.push_back(theSphere);

  }
  stk::search::coarse_search(boundingHubSphereVec_, boundingProcBoxVec_, searchMethod_, NaluEnv::self().parallel_comm(), searchKeyPair_, false);

  int iTurb=0;
  std::vector<std::pair<boundingSphere::second_type, boundingElementBox::second_type> >::const_iterator ii;
  for( ii=searchKeyPair_.begin(); ii!=searchKeyPair_.end(); ++ii ) {
    const unsigned box_proc = ii->second.proc();

    FAST.setTurbineProcNo(iTurb, box_proc);
    iTurb++;
  }



}

/** This method allocates the turbines to processors, initializes the OpenFAST instances of each turbine, populates the
 *  map of ActuatorLinePointInfo with the actuator points of all turbines and determines the elements associated with each
 *  actuator point.
*/
void
ActuatorLineFAST::initialize()
{

  allocateTurbinesToProcs();

  FAST.init() ;

  update(); // Update location of actuator points, ghosting etc.
}


/**
 * This method should be called whenever the actuator points have moved and does the following:
 *
 * + creates a new map of actuator points in ActuatorLinePointInfoMap,
 * + searches the element bounding boxes for the elements within the search radius of each
 *   actuator point,
 * + identifies the elements to be ghosted to the processor controlling the turbine,
 * + identifies the bestElem_ that contains each actuator point.
*/
void
ActuatorLineFAST::update()
{
  stk::mesh::BulkData & bulkData = realm_.bulk_data();

  // initialize need to ghost and elems to ghost
  needToGhostCount_ = 0;
  elemsToGhost_.clear();

  // clear actuatorLinePointInfoMap_
  std::map<size_t, ActuatorLineFASTPointInfo *>::iterator iterPoint;
  for( iterPoint=actuatorLinePointInfoMap_.begin(); iterPoint!=actuatorLinePointInfoMap_.end(); ++iterPoint )
    delete (*iterPoint).second;
  actuatorLinePointInfoMap_.clear();

  bulkData.modification_begin();

  if ( actuatorLineGhosting_ == NULL) {
    // create new ghosting
    std::string theGhostName = "nalu_actuator_line_ghosting";
    actuatorLineGhosting_ = &bulkData.create_ghosting( theGhostName );
  }
  else {
    bulkData.destroy_ghosting(*actuatorLineGhosting_);
  }

  bulkData.modification_end();

  // clear some of the search info
  boundingSphereVec_.clear();
  boundingElementBoxVec_.clear();
  searchKeyPair_.clear();

  // set all of the candidate elements in the search target names
  populate_candidate_elements();

  // create the ActuatorLineFASTPointInfo
  create_actuator_line_point_info_map();

  // coarse search
  determine_elems_to_ghost();

  // manage ghosting
  manage_ghosting();

  // complete filling in the set of elements connected to the centroid
  complete_search();

}


/** This function is called at each time step. This samples the velocity at each actuator point,
 *  advances the OpenFAST turbine models to Nalu's next time step and assembles the source terms
 *  in the momentum equation for Nalu.
*/
void
ActuatorLineFAST::execute()
{
  // meta/bulk data and nDim
  stk::mesh::MetaData & metaData = realm_.meta_data();
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  const int nDim = metaData.spatial_dimension();

  // extract fields
  VectorFieldType *coordinates
    = metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  VectorFieldType *velocity = metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  VectorFieldType *actuator_source
    = metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, "actuator_source");
  ScalarFieldType *actuator_source_lhs
    = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "actuator_source_lhs");
  ScalarFieldType *g
    = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "g");
  ScalarFieldType *density
    = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  ScalarFieldType *dualNodalVolume
    = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
  // deal with proper viscosity
  //  const std::string viscName = realm_.is_turbulent() ? "effective_viscosity" : "viscosity";
  //  ScalarFieldType *viscosity
  //    = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, viscName);

  // fixed size scratch
  std::vector<double> ws_pointGasVelocity(nDim);
  std::vector<double> ws_elemCentroid(nDim);
  std::vector<double> ws_pointForce(nDim);
  std::vector<double> ws_nodeForce(nDim);
  double ws_pointGasDensity;
  //  double ws_pointGasViscosity;

  // zero out source term; do this manually since there are custom ghosted entities
  stk::mesh::Selector s_nodes = stk::mesh::selectField(*actuator_source);
  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * actSrc = stk::mesh::field_data(*actuator_source, b);
    double * actSrcLhs = stk::mesh::field_data(*actuator_source_lhs, b);
    double * gF = stk::mesh::field_data(*g, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      actSrcLhs[k] = 0.0;
      gF[k] = 0.0;
      const int offSet = k*nDim;
      for ( int j = 0; j < nDim; ++j ) {
        actSrc[offSet+j] = 0.0;
      }
    }
  }

  // parallel communicate data to the ghosted elements; again can communicate points to element ranks
  if ( NULL != actuatorLineGhosting_ ) {
    std::vector< const stk::mesh::FieldBase *> ghostFieldVec;
    // fields that are needed
    ghostFieldVec.push_back(coordinates);
    ghostFieldVec.push_back(velocity);
    ghostFieldVec.push_back(dualNodalVolume);
    //    ghostFieldVec.push_back(viscosity);
    stk::mesh::communicate_field_data(*actuatorLineGhosting_, ghostFieldVec);
  }

  // loop over map and get velocity at points
  std::map<size_t, ActuatorLineFASTPointInfo *>::iterator iterPoint;
  int np=0;
  for (iterPoint  = actuatorLinePointInfoMap_.begin();
       iterPoint != actuatorLinePointInfoMap_.end();
       ++iterPoint) {

    // actuator line info object of interest
    ActuatorLineFASTPointInfo * infoObject = (*iterPoint).second;

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

    // gather nodal data to element nodes; both vector and scalar; coords are used in determinant calc
    gather_field(nDim, &ws_coordinates_[0], *coordinates, bulkData.begin_nodes(bestElem),
                 nodesPerElement);
    gather_field_for_interp(nDim, &ws_velocity_[0], *velocity, bulkData.begin_nodes(bestElem),
                            nodesPerElement);
    //    gather_field_for_interp(1, &ws_viscosity_[0], *viscosity, bulkData.begin_nodes(bestElem),
    //                            nodesPerElement);
    gather_field_for_interp(1, &ws_density_[0], *density, bulkData.begin_nodes(bestElem),
                            nodesPerElement);

    // interpolate velocity
    interpolate_field(nDim, bestElem, bulkData, infoObject->isoParCoords_.data(),
                      &ws_velocity_[0], ws_pointGasVelocity.data());

    // interpolate viscosity
    //    interpolate_field(1, bestElem, bulkData, &(infoObject->isoParCoords_[0]),
    //                      &ws_viscosity_[0], &ws_pointGasViscosity);

    // interpolate density
    interpolate_field(1, bestElem, bulkData, infoObject->isoParCoords_.data(),
                      &ws_density_[0], &ws_pointGasDensity);

    FAST.setVelocityForceNode(ws_pointGasVelocity, np, infoObject->globTurbId_);
    np = np + 1;

  }



  if ( ! FAST.isDryRun() ) {

    FAST.interpolateVel_ForceToVelNodes();

    if ( FAST.isTimeZero() ) {
      FAST.solution0();
    }

    //Step FAST
    for(int j=0; j < tStepRatio_; j++) FAST.step();
  }

  update();

  // reset the thrust and torque from each turbine to zero
  const size_t nTurbinesGlob = FAST.get_nTurbinesGlob() ;
  for(size_t iTurb=0; iTurb < nTurbinesGlob; iTurb++) {
    for (int j=0; j < nDim; j++) {
      torque[iTurb][j] = 0.0;
      thrust[iTurb][j] = 0.0;
    }
  }

  // loop over map and assemble source terms
  np = 0;
  for (iterPoint  = actuatorLinePointInfoMap_.begin();
       iterPoint != actuatorLinePointInfoMap_.end();
       ++iterPoint) {

    // actuator line info object of interest
    ActuatorLineFASTPointInfo * infoObject = (*iterPoint).second;

    FAST.getForce(ws_pointForce, np, infoObject->globTurbId_);

    std::vector<double> hubPos(3);
    std::vector<double> hubShftVec(3);
    int iTurbGlob = infoObject->globTurbId_;
    FAST.getHubPos(hubPos, iTurbGlob);
    FAST.getHubShftDir(hubShftVec, iTurbGlob);

    // get the vector of elements
    std::set<stk::mesh::Entity> nodeVec = infoObject->nodeVec_;

    switch (infoObject->nodeType_) {
    case fast::HUB:
      spread_actuator_force_to_node_vec(nDim, nodeVec, ws_pointForce, &(infoObject->centroidCoords_[0]), *coordinates, *actuator_source, *dualNodalVolume, infoObject->epsilon_, hubPos, hubShftVec, thrust[iTurbGlob], torque[iTurbGlob]);
      break;
    case fast::BLADE:
      spread_actuator_force_to_node_vec(nDim, nodeVec, ws_pointForce, &(infoObject->centroidCoords_[0]), *coordinates, *actuator_source, *dualNodalVolume, infoObject->epsilon_, hubPos, hubShftVec, thrust[iTurbGlob], torque[iTurbGlob]);
      break;
    case fast::TOWER:
      spread_actuator_force_to_node_vec(nDim, nodeVec, ws_pointForce, &(infoObject->centroidCoords_[0]), *coordinates, *actuator_source, *dualNodalVolume, infoObject->epsilon_, hubPos, hubShftVec, thrust[iTurbGlob], torque[iTurbGlob]);
      break;
    case fast::ActuatorNodeType_END:
      break;
    }

    np=np+1;
}

  if ( FAST.isDebug() ) {
    for(size_t iTurb=0; iTurb < nTurbinesGlob; iTurb++) {
      NaluEnv::self().naluOutput() << "  Thrust[" << iTurb << "] = " << thrust[iTurb][0] << " " << thrust[iTurb][1] << " " << thrust[iTurb][2] << " " << std::endl;
      NaluEnv::self().naluOutput() << "  Torque[" << iTurb << "] = " << torque[iTurb][0] << " " << torque[iTurb][1] << " " << torque[iTurb][2] << " " << std::endl;

      int processorId = FAST.get_procNo(iTurb);
      if (NaluEnv::self().parallel_rank() == processorId) {
	std::vector<double> tmpThrust(3);
	std::vector<double> tmpTorque(3);

	FAST.computeTorqueThrust(iTurb, tmpTorque, tmpThrust);

	NaluEnv::self().naluOutput() << "  Thrust ratio actual/correct = [" << thrust[iTurb][0]/tmpThrust[0] << " " << thrust[iTurb][1]/tmpThrust[1] << " " << thrust[iTurb][2]/tmpThrust[2] << "] " << std::endl;
	NaluEnv::self().naluOutput() << "  Torque ratio actual/correct = [" << torque[iTurb][0]/tmpTorque[0] << " " << torque[iTurb][1]/tmpTorque[1] << " " << torque[iTurb][2]/tmpTorque[2] << "] " << std::endl;
      }
    }
  }



  // parallel assemble (contributions from ghosted and locally owned)
  const std::vector<const stk::mesh::FieldBase*> sumFieldVec(1, actuator_source);
  stk::mesh::parallel_sum_including_ghosts(bulkData, sumFieldVec);

  const std::vector<const stk::mesh::FieldBase*> sumFieldG(1, g);
  stk::mesh::parallel_sum_including_ghosts(bulkData, sumFieldG);


}

// creates bounding boxes around elements belonging to the searchParts
void
ActuatorLineFAST::populate_candidate_elements()
{
  stk::mesh::MetaData & metaData = realm_.meta_data();
  stk::mesh::BulkData & bulkData = realm_.bulk_data();

  const int nDim = metaData.spatial_dimension();

  // fields
  VectorFieldType *coordinates = metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // point data structures
  Point minCorner, maxCorner;

  // extract part
  stk::mesh::PartVector searchParts;
  for ( size_t k = 0; k < searchTargetNames_.size(); ++k ) {
    stk::mesh::Part *thePart = metaData.get_part(searchTargetNames_[k]);
    if ( NULL != thePart )
      searchParts.push_back(thePart);
    else
      throw std::runtime_error("ActuatorLineFAST: Part is null" + searchTargetNames_[k]);
  }

  // selector and bucket loop
  stk::mesh::Selector s_locally_owned = metaData.locally_owned_part()
    &stk::mesh::selectUnion(searchParts);

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned );

  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
        ib != elem_buckets.end() ; ++ib ) {

    stk::mesh::Bucket & b = **ib;

    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get element
      stk::mesh::Entity elem = b[k];

      // initialize max and min
      for (int j = 0; j < nDim; ++j ) {
        minCorner[j] = +1.0e16;
        maxCorner[j] = -1.0e16;
      }

      // extract elem_node_relations
      stk::mesh::Entity const* elem_node_rels = bulkData.begin_nodes(elem);
      const int num_nodes = bulkData.num_nodes(elem);

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = elem_node_rels[ni];

        // pointers to real data
        const double * coords = stk::mesh::field_data(*coordinates, node );

        // check max/min
        for ( int j = 0; j < nDim; ++j ) {
          minCorner[j] = std::min(minCorner[j], coords[j]);
          maxCorner[j] = std::max(maxCorner[j], coords[j]);
        }
      }

      // setup ident
      stk::search::IdentProc<uint64_t,int> theIdent(bulkData.identifier(elem), NaluEnv::self().parallel_rank());

      // create the bounding point box and push back
      boundingElementBox theBox(Box(minCorner,maxCorner), theIdent);
      boundingElementBoxVec_.push_back(theBox);
    }
  }

}

// Creates bounding boxes around the subdomain of each processor
void
ActuatorLineFAST::populate_candidate_procs()
{
  stk::mesh::MetaData & metaData = realm_.meta_data();

  const int nDim = metaData.spatial_dimension();

  // fields
  VectorFieldType *coordinates = metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // point data structures
  //  Point minCorner, maxCorner;
  std::vector<Point> minCorner(1), maxCorner(1);
  std::vector<Point> gMinCorner, gMaxCorner;

  // initialize max and min
  for (int j = 0; j < nDim; ++j ) {
    minCorner[0][j] = +1.0e16;
    maxCorner[0][j] = -1.0e16;
  }

  // extract part
  stk::mesh::PartVector searchParts;
  for ( size_t k = 0; k < searchTargetNames_.size(); ++k ) {
    stk::mesh::Part *thePart = metaData.get_part(searchTargetNames_[k]);
    if ( NULL != thePart )
      searchParts.push_back(thePart);
    else
      throw std::runtime_error("ActuatorLineFAST: Part is null" + searchTargetNames_[k]);
  }

  // selector and bucket loop
  stk::mesh::Selector s_locally_owned = metaData.locally_owned_part()
    &stk::mesh::selectUnion(searchParts);

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_locally_owned );

  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
        ib != elem_buckets.end() ; ++ib ) {

    stk::mesh::Bucket & b = **ib;

    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get element
      stk::mesh::Entity node = b[k];

      // pointers to real data
      const double * coords = stk::mesh::field_data(*coordinates, node );

      // check max/min
      for ( int j = 0; j < nDim; ++j ) {
	minCorner[0][j] = std::min(minCorner[0][j], coords[j]);
	maxCorner[0][j] = std::max(maxCorner[0][j], coords[j]);
      }

    }
  }

  stk::parallel_vector_concat(NaluEnv::self().parallel_comm(), minCorner, gMinCorner);
  stk::parallel_vector_concat(NaluEnv::self().parallel_comm(), maxCorner, gMaxCorner);

  for(int j = 0; j < NaluEnv::self().parallel_size(); j++) {
    // setup ident
    stk::search::IdentProc<uint64_t,int> theIdent(j, j);

    // create the bounding point box and push back
    boundingElementBox theBox(Box(gMinCorner[j],gMaxCorner[j]), theIdent);
    boundingProcBoxVec_.push_back(theBox);

  }

}

// Determine elements to be ghosted to the owning rank of the turbine
void
ActuatorLineFAST::determine_elems_to_ghost()
{
  stk::mesh::BulkData & bulkData = realm_.bulk_data();

  stk::search::coarse_search(boundingSphereVec_, boundingElementBoxVec_, searchMethod_,
                             NaluEnv::self().parallel_comm(), searchKeyPair_);

  // lowest effort is to ghost elements to the owning rank of the point; can just as easily do the opposite
  std::vector<std::pair<boundingSphere::second_type, boundingElementBox::second_type> >::const_iterator ii;
  for( ii=searchKeyPair_.begin(); ii!=searchKeyPair_.end(); ++ii ) {

    const uint64_t theBox = ii->second.id();
    unsigned theRank = NaluEnv::self().parallel_rank();
    const unsigned pt_proc = ii->first.proc();
    const unsigned box_proc = ii->second.proc();
    if ( (box_proc == theRank) && (pt_proc != theRank) ) {

      // Send box to pt proc

      // find the element
      stk::mesh::Entity theElemMeshObj = bulkData.get_entity(stk::topology::ELEMENT_RANK, theBox);
      if ( !(bulkData.is_valid(theElemMeshObj)) )
        throw std::runtime_error("no valid entry for element");

      // new element to ghost counter
      needToGhostCount_++;

      // deal with elements to push back to be ghosted
      stk::mesh::EntityProc theElemPair(theElemMeshObj, pt_proc);
      elemsToGhost_.push_back(theElemPair);
    }
  }

}

//--------------------------------------------------------------------------
//-------- create_actuator_line_point_info_map -----------------------------
//--------------------------------------------------------------------------
void
ActuatorLineFAST::create_actuator_line_point_info_map() {

  stk::mesh::MetaData & metaData = realm_.meta_data();
  const int nDim = metaData.spatial_dimension();

  size_t np = 0;

  for ( size_t iTurb = 0; iTurb < actuatorLineInfo_.size(); ++iTurb ) {

    const ActuatorLineFASTInfo *actuatorLineInfo = actuatorLineInfo_[iTurb];

    int processorId = FAST.get_procNo(iTurb);
    if ( processorId == NaluEnv::self().parallel_rank() ) {

      // define a point that will hold the centroid
      Point centroidCoords;

      // scratch array for coordinates and dummy array for velocity
      std::vector<double> currentCoords (3,0.0);

      // loop over all points for this turbine
      const int numForcePts = FAST.get_numForcePts(iTurb); // Total number of elements
      //      actuatorLineInfo->numPoints_ = numVelPts ; // Can't change a const pointer

      if (! FAST.isDryRun() ) {
	for(int iNode = 0; iNode < numForcePts; iNode++) {
	  stk::search::IdentProc<uint64_t,int> theIdent(np, NaluEnv::self().parallel_rank());

	  // set model coordinates from FAST
	  // move the coordinates; set the velocity... may be better on the lineInfo object
	  FAST.getForceNodeCoordinates(currentCoords, np, iTurb);

          double searchRadius = actuatorLineInfo->epsilon_.x_ * sqrt(log(1.0/0.001));

	  for ( int j = 0; j < nDim; ++j )
	    centroidCoords[j] = currentCoords[j];

          // create the bounding point sphere and push back
          boundingSphere theSphere( Sphere(centroidCoords, searchRadius), theIdent);
          boundingSphereVec_.push_back(theSphere);

          // create the point info and push back to map
          Coordinates epsilon;
          switch (FAST.getForceNodeType(iTurb, np)) {
          case fast::HUB: {
              // Calculate epsilon for hub node based on cd and area here
              float nac_area = FAST.get_nacelleArea(iTurb);
              float nac_cd = FAST.get_nacelleCd(iTurb);

              // The constant pi
              const float pi = acos(-1.0);

              for (int j=0; j<nDim; j++) {


                  // Compute epsilon only if drag coefficient is greater than zero
                  if (nac_cd>0){
                      // This model is used to set the momentum thickness
                      // of the wake (Martinez-Tossas PhD Thesis 2017)
                      float tmpEps = std::sqrt(2.0 / pi * nac_cd * nac_area); 
                      epsilon.x_ = tmpEps;
                      epsilon.y_ = tmpEps;
                      epsilon.z_ = tmpEps;
                  }
                  
                  // If no nacelle force just specify the standard value
                  // (it will not be used)
                  else{
                      epsilon = actuatorLineInfo->epsilon_;
                      //~ epsilon.y_ = actuatorLineInfo->epsilon_;
                      //~ epsilon.z_ = actuatorLineInfo->epsilon_;
                      }

              }
              break;
          }
          case fast::BLADE:
              epsilon = actuatorLineInfo->epsilon_;
              break;
          case fast::TOWER:
              epsilon = actuatorLineInfo->epsilon_;
              break;
          default: throw std::runtime_error("Actuator line model node type not valid"); 
              break;
          }
          
	  ActuatorLineFASTPointInfo *actuatorLinePointInfo
	    = new ActuatorLineFASTPointInfo(iTurb, centroidCoords,
					    searchRadius, epsilon,
					    FAST.getForceNodeType(iTurb, np));
	  actuatorLinePointInfoMap_[np] = actuatorLinePointInfo;

	  np=np+1;
	}

      }
      else {
	NaluEnv::self().naluOutput() << "Proc " << NaluEnv::self().parallel_rank() << " glob iTurb " << iTurb << std::endl ;
      }

    }

  }

}


//--------------------------------------------------------------------------
//-------- manage_ghosting -------------------------------------------------
//--------------------------------------------------------------------------
void
ActuatorLineFAST::manage_ghosting()
{
  stk::mesh::BulkData & bulkData = realm_.bulk_data();

  // check for ghosting need
  uint64_t g_needToGhostCount = 0;
  stk::all_reduce_sum(NaluEnv::self().parallel_comm(), &needToGhostCount_, &g_needToGhostCount, 1);
  if (g_needToGhostCount > 0) {
    NaluEnv::self().naluOutputP0() << "ActuatorLineFAST alg will ghost a number of entities for velocity actuator nodes: "
                                   << g_needToGhostCount  << std::endl;
    bulkData.modification_begin();
    bulkData.change_ghosting( *actuatorLineGhosting_, elemsToGhost_);
    bulkData.modification_end();
  }
  else {
    NaluEnv::self().naluOutputP0() << "ActuatorLineFAST alg will NOT ghost entities for velcity actuator nodes: " << std::endl;
  }

}


//--------------------------------------------------------------------------
//-------- complete_search -------------------------------------------------
//--------------------------------------------------------------------------
void
ActuatorLineFAST::complete_search()
{
  stk::mesh::MetaData & metaData = realm_.meta_data();
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  const int nDim = metaData.spatial_dimension();

  // extract fields
  VectorFieldType *coordinates
    = metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // now proceed with the standard search
  std::vector<std::pair<boundingSphere::second_type, boundingElementBox::second_type> >::const_iterator ii;
  for( ii=searchKeyPair_.begin(); ii!=searchKeyPair_.end(); ++ii ) {

    const uint64_t thePt = ii->first.id();
    const uint64_t theBox = ii->second.id();
    const unsigned theRank = NaluEnv::self().parallel_rank();
    const unsigned pt_proc = ii->first.proc();

    // check if I own the point...
    if ( theRank == pt_proc ) {

      // yes, I own the point...

      // proceed as required; all elements should have already been ghosted via the coarse search
      stk::mesh::Entity elem = bulkData.get_entity(stk::topology::ELEMENT_RANK, theBox);
      if ( !(bulkData.is_valid(elem)) )
        throw std::runtime_error("no valid entry for element");

      // find the point data structure
      std::map<size_t, ActuatorLineFASTPointInfo *>::iterator iterPoint;
      iterPoint=actuatorLinePointInfoMap_.find(thePt);
      if ( iterPoint == actuatorLinePointInfoMap_.end() )
        throw std::runtime_error("no valid entry for actuatorLinePointInfoMap_");

      // extract the point object and push back the element to either the best
      // candidate or the standard vector of elements
      ActuatorLineFASTPointInfo *actuatorLinePointInfo = iterPoint->second;

      // extract topo and master element for this topo
      const stk::mesh::Bucket &theBucket = bulkData.bucket(elem);
      const stk::topology &elemTopo = theBucket.topology();
      MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(elemTopo);
      const int nodesPerElement = meSCS->nodesPerElement_;

      // gather elemental coords
      std::vector<double> elementCoords(nDim*nodesPerElement);
      gather_field_for_interp(nDim, &elementCoords[0], *coordinates, bulkData.begin_nodes(elem),
                   nodesPerElement);


      // find isoparametric points
      std::vector<double> isoParCoords(nDim);
      const double nearestDistance = meSCS->isInElement(&elementCoords[0],
                                                        &(actuatorLinePointInfo->centroidCoords_[0]),
                                                        &(isoParCoords[0]));


      // save off best element and its isoparametric coordinates for this point
      if ( nearestDistance < actuatorLinePointInfo->bestX_ ) {
        actuatorLinePointInfo->bestX_ = nearestDistance;
        actuatorLinePointInfo->isoParCoords_ = isoParCoords;
        actuatorLinePointInfo->bestElem_ = elem;
      }
        // extract elem_node_relations
        stk::mesh::Entity const* elem_node_rels = bulkData.begin_nodes(elem);
        const unsigned num_nodes = bulkData.num_nodes(elem);
        for (unsigned inode = 0; inode < num_nodes; inode++) {
            stk::mesh::Entity node = elem_node_rels[inode];
            actuatorLinePointInfo->nodeVec_.insert(node);
        }
    }
    else {
      // not this proc's issue
    }
  }

}


//--------------------------------------------------------------------------
//-------- resize_std_vector -----------------------------------------------
//--------------------------------------------------------------------------
void
ActuatorLineFAST::resize_std_vector(
  const int &sizeOfField,
  std::vector<double> &theVector,
  stk::mesh::Entity elem,
  const stk::mesh::BulkData & bulkData)
{
  const stk::topology &elemTopo = bulkData.bucket(elem).topology();
  MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(elemTopo);
  const int nodesPerElement = meSCS->nodesPerElement_;
  theVector.resize(nodesPerElement*sizeOfField);
}


//--------------------------------------------------------------------------
//-------- gather_field ----------------------------------------------------
//--------------------------------------------------------------------------
void
ActuatorLineFAST::gather_field(
  const int &sizeOfField,
  double *fieldToFill,
  const stk::mesh::FieldBase &stkField,
  stk::mesh::Entity const* elem_node_rels,
  const int &nodesPerElement)
{
  for ( int ni = 0; ni < nodesPerElement; ++ni ) {
    stk::mesh::Entity node = elem_node_rels[ni];
    const double * theField = (double*)stk::mesh::field_data(stkField, node );
    for ( int j = 0; j < sizeOfField; ++j ) {
      const int offSet = ni*sizeOfField+j;
      fieldToFill[offSet] = theField[j];
    }
  }
}


//--------------------------------------------------------------------------
//-------- gather_field_for_interp -----------------------------------------
//--------------------------------------------------------------------------
void
ActuatorLineFAST::gather_field_for_interp(
  const int &sizeOfField,
  double *fieldToFill,
  const stk::mesh::FieldBase &stkField,
  stk::mesh::Entity const* elem_node_rels,
  const int &nodesPerElement)
{
  for ( int ni = 0; ni < nodesPerElement; ++ni ) {
    stk::mesh::Entity node = elem_node_rels[ni];
    const double * theField = (double*)stk::mesh::field_data(stkField, node );
    for ( int j = 0; j < sizeOfField; ++j ) {
      const int offSet = j*nodesPerElement + ni;
      fieldToFill[offSet] = theField[j];
    }
  }
}


//--------------------------------------------------------------------------
//-------- compute_volume --------------------------------------------------
//--------------------------------------------------------------------------
double
ActuatorLineFAST::compute_volume(
  const int &nDim,
  stk::mesh::Entity elem,
  const stk::mesh::BulkData & bulkData)
{
  // extract master element from the bucket in which the element resides
  const stk::topology &elemTopo = bulkData.bucket(elem).topology();
  MasterElement *meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(elemTopo);
  const int numScvIp = meSCV->numIntPoints_;

  // compute scv for this element
  ws_scv_volume_.resize(numScvIp);
  double scv_error = 0.0;
  meSCV->determinant(1, &ws_coordinates_[0], &ws_scv_volume_[0], &scv_error);

  double elemVolume = 0.0;
  for ( int ip = 0; ip < numScvIp; ++ip ) {
    elemVolume += ws_scv_volume_[ip];
  }
  return elemVolume;
}


//--------------------------------------------------------------------------
//-------- interpolate_field -----------------------------------------------
//--------------------------------------------------------------------------
void
ActuatorLineFAST::interpolate_field(
  const int &sizeOfField,
  stk::mesh::Entity elem,
  const stk::mesh::BulkData & bulkData,
  double *isoParCoords,
  const double *fieldAtNodes,
  double *pointField)
{
  // extract master element from the bucket in which the element resides
  const stk::topology &elemTopo = bulkData.bucket(elem).topology();
  MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(elemTopo);

  // interpolate velocity to this best point
  meSCS->interpolatePoint(
    sizeOfField,
    isoParCoords,
    fieldAtNodes,
    pointField);
}


//--------------------------------------------------------------------------
//-------- compute_elem_centroid -------------------------------------------
//--------------------------------------------------------------------------
void
ActuatorLineFAST::compute_elem_centroid(
  const int &nDim,
  double *elemCentroid,
  const int & nodesPerElement)
{
  // zero
  for ( int j = 0; j < nDim; ++j )
    elemCentroid[j] = 0.0;

  // assemble
  for ( int ni = 0; ni < nodesPerElement; ++ni ) {
    for ( int j=0; j < nDim; ++j ) {
      elemCentroid[j] += ws_coordinates_[ni*nDim+j]/nodesPerElement;
    }
  }
}


//--------------------------------------------------------------------------
//-------- compute_distance ------------------------------------------------
//--------------------------------------------------------------------------
double
ActuatorLineFAST::compute_distance(
  const int &nDim,
  const double *elemCentroid,
  const double *pointCentroid)
{
  double distance = 0.0;
  for ( int j = 0; j < nDim; ++j )
    distance += std::pow(elemCentroid[j] - pointCentroid[j], 2);
  distance = std::sqrt(distance);
  return distance;
}


// Spread actuator force to nodes
void
ActuatorLineFAST::spread_actuator_force_to_node_vec(
  const int &nDim,
  std::set<stk::mesh::Entity>& nodeVec,
  const std::vector<double>& actuator_force,
  const double * actuator_node_coordinates,
  const stk::mesh::FieldBase & coordinates,
  stk::mesh::FieldBase & actuator_source,
  const stk::mesh::FieldBase & dual_nodal_volume,
  const Coordinates & epsilon,
  const std::vector<double> & hubPt,
  const std::vector<double> & hubShftDir,
  std::vector<double> & thr,
  std::vector<double> & tor
)
{
    std::vector<double> ws_nodeForce(nDim);

    std::set<stk::mesh::Entity>::iterator iNode;
    for (iNode = nodeVec.begin(); iNode != nodeVec.end(); ++iNode ) {

      stk::mesh::Entity node = *iNode;
      const double * node_coords = (double*)stk::mesh::field_data(coordinates, node );
      const double * dVol = (double*)stk::mesh::field_data(dual_nodal_volume, node );
      // compute distance
      const double distance = compute_distance(nDim, node_coords, actuator_node_coordinates);
      // project the force to this node with projection function
      double gA = isotropic_Gaussian_projection(nDim, distance, epsilon);
      compute_node_force_given_weight(nDim, gA, &actuator_force[0], &ws_nodeForce[0]);
      double * sourceTerm = (double*)stk::mesh::field_data(actuator_source, node );
      for ( int j=0; j < nDim; ++j ) sourceTerm[j] += ws_nodeForce[j];

      add_thrust_torque_contrib(nDim, node_coords, *dVol, ws_nodeForce, hubPt, hubShftDir, thr, tor);

    }

}

// Add to thrust and torque contribution from current node
void
ActuatorLineFAST::add_thrust_torque_contrib(
    const int &nDim,
    const double * nodeCoords,
    const double dVol,
    const std::vector<double> & nodeForce,
    const std::vector<double> & hubPt,
    const std::vector<double> & hubShftDir,
    std::vector<double> & thr,
    std::vector<double> & tor
    )
{

    std::vector<double> rPerpShft(nDim);
    std::vector<double> r(nDim);
    for (int j=0; j < nDim; j++) {
        r[j] = nodeCoords[j] - hubPt[j];
    }
    double rDotHubShftVec = r[0]*hubShftDir[0] + r[1]*hubShftDir[1] + r[2]*hubShftDir[2];
    for (int j=0; j < nDim; j++) {
        rPerpShft[j] = r[j] - rDotHubShftVec * hubShftDir[j];
    }

    for (int j=0; j < nDim; j++) {
        thr[j] += nodeForce[j] * dVol;
    }
    tor[0] += (rPerpShft[1]*nodeForce[2] - rPerpShft[2]*nodeForce[1]) * dVol;
    tor[1] += (rPerpShft[2]*nodeForce[0] - rPerpShft[0]*nodeForce[2]) * dVol;
    tor[2] += (rPerpShft[0]*nodeForce[1] - rPerpShft[1]*nodeForce[0]) * dVol;

}

} // namespace nalu
} // namespace Sierra

#endif
