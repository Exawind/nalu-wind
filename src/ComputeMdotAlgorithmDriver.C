/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include "ComputeMdotAlgorithmDriver.h"
#include "Algorithm.h"
#include "AlgorithmDriver.h"
#include "FieldTypeDef.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

namespace sierra{
namespace nalu{

class Realm;

//==========================================================================
// Class Definition
//==========================================================================
// ComputeMdotAlgorithmDriver - Drives mdot 
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeMdotAlgorithmDriver::ComputeMdotAlgorithmDriver(
  Realm &realm)
  : AlgorithmDriver(realm),
    solnOpts_(*realm.solutionOptions_),
    hasMass_(false),
    lumpedMass_(true)
{
  // look for "density_time_derivative" from element_source_terms; must be look ahead
  std::vector<const YAML::Node*> elemSrcTermsNode;
  NaluParsingHelper::find_nodes_given_key("element_source_terms", realm_.node_, elemSrcTermsNode);
  if ( elemSrcTermsNode.size() > 0 ) {
    std::map<std::string, std::vector<std::string> > elemSrcTermsMap;
    const YAML::Node y_solution_options = expect_map(realm_.node_,"solution_options", false);
    const YAML::Node y_options = expect_sequence(y_solution_options, "options", true);
    for (size_t ioption = 0; ioption < y_options.size(); ++ioption) {
      const YAML::Node y_option = y_options[ioption] ;
      if ( expect_map(y_option, "element_source_terms", true) ) {
        const YAML::Node ySrc = y_option["element_source_terms"];
        ySrc >> elemSrcTermsMap;
        std::map<std::string, std::vector<std::string> >::iterator isrc
          = elemSrcTermsMap.find("continuity");
        if ( isrc != elemSrcTermsMap.end() ) {          
          std::vector<std::string> mapNameVec = isrc->second;
          for (size_t k = 0; k < mapNameVec.size(); ++k ) {
            std::string srcName = mapNameVec[k];
            if (srcName == "density_time_derivative" ) {
              hasMass_ = true;
              lumpedMass_ = false;
            }
            else if (srcName == "lumped_density_time_derivative" ) {
              hasMass_ = true;
              lumpedMass_ = true;
            }
          }
        }
      }
    }
  }

  // look for "density_time_derivative" from source_terms; must be look ahead
  std::vector<const YAML::Node*> nodalSrcTermsNode;
  NaluParsingHelper::find_nodes_given_key("source_terms", realm_.node_, nodalSrcTermsNode);
  if ( nodalSrcTermsNode.size() > 0 ) {
    std::map<std::string, std::vector<std::string> > nodalSrcTermsMap;
    const YAML::Node y_solution_options = expect_map(realm_.node_,"solution_options", false);
    const YAML::Node y_options = expect_sequence(y_solution_options, "options", true);
    for (size_t ioption = 0; ioption < y_options.size(); ++ioption) {
      const YAML::Node y_option = y_options[ioption] ;
      if ( expect_map(y_option, "source_terms", true) ) {
        const YAML::Node ySrc = y_option["source_terms"];
        ySrc >> nodalSrcTermsMap;
        std::map<std::string, std::vector<std::string> >::iterator isrc
          = nodalSrcTermsMap.find("continuity");
        if ( isrc != nodalSrcTermsMap.end() ) {          
          std::vector<std::string> mapNameVec = isrc->second;
          for (size_t k = 0; k < mapNameVec.size(); ++k ) {
            std::string srcName = mapNameVec[k];
            if (srcName == "density_time_derivative" ) {
              hasMass_ = true;
              lumpedMass_ = true;
            }
          }
        }
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ComputeMdotAlgorithmDriver::~ComputeMdotAlgorithmDriver()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- pre_work --------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMdotAlgorithmDriver::pre_work()
{
  // set post processing to zero
  solnOpts_.mdotAlgAccumulation_ = 0.0;
  solnOpts_.mdotAlgInflow_ = 0.0;
  solnOpts_.mdotAlgOpen_ = 0.0;

  // also global correction algorithm
  solnOpts_.mdotAlgOpenCorrection_ = 0.0;
  solnOpts_.mdotAlgOpenIpCount_= 0.0;
}

//--------------------------------------------------------------------------
//-------- post_work -------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMdotAlgorithmDriver::post_work()
{
  // compute drho/dt * scv * dt 
  double accumulation = hasMass_ ? compute_accumulation() : 0.0;
  
  // parallel communicate; mdot and accumulation; mdot calculations in execuate provided these values
  double l_sum[3] = {accumulation, solnOpts_.mdotAlgInflow_, solnOpts_.mdotAlgOpen_};
  double g_sum[3] = {};
  stk::ParallelMachine comm = NaluEnv::self().parallel_comm();
  stk::all_reduce_sum(comm, l_sum, g_sum, 3);

  // set parameters for later usage
  solnOpts_.mdotAlgAccumulation_ = g_sum[0];
  solnOpts_.mdotAlgInflow_ = g_sum[1];
  solnOpts_.mdotAlgOpen_ = g_sum[2];

  // deal with global correction algorithm
  if ( solnOpts_.activateOpenMdotCorrection_ ) {
    size_t l_ip = solnOpts_.mdotAlgOpenIpCount_;
    size_t g_ip = 0;
    stk::all_reduce_sum(comm, &l_ip, &g_ip, 1);
    solnOpts_.mdotAlgOpenIpCount_ = g_ip;
    const double finalCorrection = (g_sum[0] + g_sum[1] + g_sum[2])/g_ip;
    solnOpts_.mdotAlgOpenCorrection_ = finalCorrection;
    solnOpts_.mdotAlgOpenIpCount_ = g_ip;
    correct_open_mdot(finalCorrection);
  }
}

//--------------------------------------------------------------------------
//-------- compute_accumulation --------------------------------------------
//--------------------------------------------------------------------------
double
ComputeMdotAlgorithmDriver::compute_accumulation()
{
  stk::mesh::MetaData & metaData = realm_.meta_data();

  const double dt = realm_.get_time_step();
  const int nDim = metaData.spatial_dimension();

  // initialize accumulation term to zero
  double accumulation = 0.0;

  // extract time parameters
  const double gamma1 = realm_.get_gamma1();
  const double gamma2 = realm_.get_gamma2();
  const double gamma3 = realm_.get_gamma3(); // gamma3 may be zero

  // extract fields
  ScalarFieldType *density = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "density");
  ScalarFieldType &densityNp1 = density->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &densityN = density->field_of_state(stk::mesh::StateN);
  ScalarFieldType &densityNm1 = (density->number_of_states() == 2) 
    ? density->field_of_state(stk::mesh::StateN) : density->field_of_state(stk::mesh::StateNM1);
  VectorFieldType *coordinates = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, solnOpts_.get_coordinates_name());

  //  required space
  std::vector<double> ws_shape_function;
  std::vector<double> ws_rhoNp1;
  std::vector<double> ws_rhoN;
  std::vector<double> ws_rhoNm1;
  std::vector<double> ws_coordinates;
  std::vector<double> ws_scv_volume;

  // selector (everywhere density lives, locally owned and active) 
  stk::mesh::Selector s_locally_owned = stk::mesh::selectField(*density)    
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned);
  
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin() ;
        ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    // extract master element
    MasterElement *meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(b.topology());

    // extract master element specifics
    const int nodesPerElement = meSCV->nodesPerElement_;
    const int numScvIp = meSCV->num_integration_points();

    // resize
    ws_shape_function.resize(numScvIp*nodesPerElement);
    ws_rhoNp1.resize(nodesPerElement);
    ws_rhoN.resize(nodesPerElement);
    ws_rhoNm1.resize(nodesPerElement);
    ws_coordinates.resize(nDim*nodesPerElement);
    ws_scv_volume.resize(numScvIp);
    
    if ( lumpedMass_ )
      meSCV->shifted_shape_fcn(&ws_shape_function[0]);
    else
      meSCV->shape_fcn(&ws_shape_function[0]);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      //===============================================
      // gather nodal data; this is how we do it now..
      //===============================================
      stk::mesh::Entity const *  node_rels = b.begin_nodes(k);
      int num_nodes = b.num_nodes(k);

      // sanity check on num nodes
      ThrowAssert( num_nodes == nodesPerElement );

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];

        // pointers to real data
        const double * coords = stk::mesh::field_data(*coordinates, node );

        // gather scalars
        ws_rhoNp1[ni]  = *stk::mesh::field_data(densityNp1, node );
        ws_rhoN[ni]  = *stk::mesh::field_data(densityN, node );
        ws_rhoNm1[ni]  = *stk::mesh::field_data(densityNm1, node );

        // gather vectors
        const int niNdim = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          ws_coordinates[niNdim+j] = coords[j];
        }
      }

      // compute geometry
      double scv_error = 0.0;
      meSCV->determinant(1, &ws_coordinates[0], &ws_scv_volume[0], &scv_error);

      for ( int ip = 0; ip < numScvIp; ++ip ) {

        // zero out; scalar
        double rhoNm1Scv = 0.0;
        double rhoNScv = 0.0;
        double rhoNp1Scv = 0.0;
        
        const int offSet = ip*nodesPerElement;
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          // save off shape function
          const double r = ws_shape_function[offSet+ic];
          
          // density
          rhoNm1Scv += r*ws_rhoNm1[ic];
          rhoNScv += r*ws_rhoN[ic];
          rhoNp1Scv += r*ws_rhoNp1[ic];
        }
        
        accumulation +=  
          (gamma1*rhoNp1Scv + gamma2*rhoNScv + gamma3*rhoNm1Scv)/dt*ws_scv_volume[ip];
      }
    }
  }
  return accumulation;
}

//--------------------------------------------------------------------------
//-------- correct_open_mdot -----------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMdotAlgorithmDriver::correct_open_mdot(const double finalCorrection)
{
  // extract field
  stk::mesh::MetaData & metaData = realm_.meta_data();
  GenericFieldType *openMassFlowRate = metaData.get_field<GenericFieldType>(metaData.side_rank(), "open_mass_flow_rate");
  if ( NULL != openMassFlowRate ) {
    
    double mdotSum = 0.0;

    // selector (everywhere density lives, locally owned and active) 
    stk::mesh::Selector s_locally_owned = stk::mesh::selectField(*openMassFlowRate)    
      & !(realm_.get_inactive_selector());
    
    stk::mesh::BucketVector const& face_buckets =
      realm_.get_buckets( metaData.side_rank(), s_locally_owned );
    for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
          ib != face_buckets.end() ; ++ib ) {
      stk::mesh::Bucket & b = **ib ;
      
      // face master element
      MasterElement *meFC = MasterElementRepo::get_surface_master_element(b.topology());
      const int numScsBip = meFC->num_integration_points();
      
      const stk::mesh::Bucket::size_type length   = b.size();
      
      for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
        
        double * mdot    = stk::mesh::field_data(*openMassFlowRate, b, k);

        // loop over boundary ips and correct open mdot
        for ( int ip = 0; ip < numScsBip; ++ip ) {
          mdot[ip] -= finalCorrection;
          mdotSum += mdot[ip];
        }
      }
    }

    // provide post corrected mdot
    double g_mdotSum = 0.0;
    stk::ParallelMachine comm = NaluEnv::self().parallel_comm();
    stk::all_reduce_sum(comm, &mdotSum, &g_mdotSum, 1);
    solnOpts_.mdotAlgOpenPost_ = g_mdotSum;
  }
}

//--------------------------------------------------------------------------
//-------- provide_output -----------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMdotAlgorithmDriver::provide_output()
{
  // output mass closure
  const double integratedAccumulation = solnOpts_.mdotAlgAccumulation_;
  const double integratedInflow = solnOpts_.mdotAlgInflow_ ;
  const double integratedOpen = solnOpts_.mdotAlgOpen_;
  const double totalMassClosure = integratedAccumulation + integratedInflow + integratedOpen;
  NaluEnv::self().naluOutputP0() << "Mass Balance Review:  " << std::endl;
  NaluEnv::self().naluOutputP0() << "Density accumulation: " << integratedAccumulation << std::endl;
  NaluEnv::self().naluOutputP0() << "Integrated inflow:    " << std::setprecision (16) << integratedInflow << std::endl;
  NaluEnv::self().naluOutputP0() << "Integrated open:      " << std::setprecision (16) << integratedOpen << std::endl;
  NaluEnv::self().naluOutputP0() << "Total mass closure:   " << std::setprecision (6) << totalMassClosure << std::endl;
  if ( solnOpts_.activateOpenMdotCorrection_ ) {
    const size_t ipCount = solnOpts_.mdotAlgOpenIpCount_;
    const double ipCorrection = solnOpts_.mdotAlgOpenCorrection_;
    NaluEnv::self().naluOutputP0() << "A mass correction of: " << ipCorrection
                                   << " occurred on: " << ipCount
                                   << " boundary integration points: "  << std::endl;
    NaluEnv::self().naluOutputP0() << "Post-corrected integrated open: " << std::setprecision (16) << solnOpts_.mdotAlgOpenPost_ << std::endl;
  }
}

} // namespace nalu
} // namespace Sierra
