/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <SolutionNormPostProcessing.h>
#include <AuxFunctionAlgorithm.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>
#include <Realm.h>

// the factory of aux functions
#include <user_functions/SteadyThermal3dContactAuxFunction.h>
#include <user_functions/SteadyThermal3dContactDtDxAuxFunction.h>
#include <user_functions/SteadyThermalContactAuxFunction.h>
#include <user_functions/SteadyTaylorVortexVelocityAuxFunction.h>
#include <user_functions/SteadyTaylorVortexGradPressureAuxFunction.h>
#include <user_functions/ConvectingTaylorVortexVelocityAuxFunction.h>
#include <user_functions/ConvectingTaylorVortexPressureAuxFunction.h>
#include <user_functions/VariableDensityVelocityAuxFunction.h>
#include <user_functions/VariableDensityNonIsoTemperatureAuxFunction.h>
#include <user_functions/BoussinesqNonIsoVelocityAuxFunction.h>
#include <user_functions/BoussinesqNonIsoTemperatureAuxFunction.h>
#include <user_functions/VariableDensityMixFracAuxFunction.h>
#include <user_functions/KovasznayVelocityAuxFunction.h>
#include <user_functions/KovasznayPressureAuxFunction.h>
#include <user_functions/WindEnergyTaylorVortexAuxFunction.h>
#include <user_functions/WindEnergyTaylorVortexPressureAuxFunction.h>

#include <user_functions/OneTwoTenVelocityAuxFunction.h>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// basic c++
#include <stdexcept>
#include <string>
#include <fstream>
#include <iomanip>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// SolutionNormPostProcessing - norm solution post processing; all nodal
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
SolutionNormPostProcessing::SolutionNormPostProcessing(
  Realm &realm,
  const YAML::Node &node)
  : realm_(realm),
    outputFrequency_(100),
    totalDofCompSize_(0),
    outputFileName_("norms.dat"),
    w_(12),
    percision_(6)
{
  // load the data
  load(node);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
SolutionNormPostProcessing::~SolutionNormPostProcessing()
{
  // clean-up; aux function algorithm deletes aux function 
  std::vector<AuxFunctionAlgorithm *>::iterator ii;
  for( ii=populateExactNodalFieldAlg_.begin(); ii!=populateExactNodalFieldAlg_.end(); ++ii )
    delete *ii;
}

//--------------------------------------------------------------------------
//-------- load ------------------------------------------------------------
//--------------------------------------------------------------------------
void
SolutionNormPostProcessing::load(
  const YAML::Node & y_node)
{
  // output for results
  const YAML::Node y_norm = y_node["solution_norm"];
  if (y_norm)
  {    
    // output frequency
    get_if_present(y_norm, "output_frequency", outputFrequency_, outputFrequency_);

    // output name
    get_if_present(y_norm, "file_name", outputFileName_, outputFileName_);

    // spacing
    get_if_present(y_norm, "spacing", w_, w_);

    // percision
    get_if_present(y_norm, "percision", percision_, percision_);
    get_if_present(y_norm, "precision", percision_, percision_);


    // target matches the physics description (see Material model)
    
    // find the pair; create some space for the names
    const YAML::Node y_dof_pair = y_norm["dof_user_function_pair"];
    std::string dofName, functionName;
    if (y_dof_pair)
    {
      for (size_t ioption = 0; ioption < y_dof_pair.size(); ++ioption) {
        const YAML::Node y_var = y_dof_pair[ioption];
        size_t varPairSize = y_var.size();
        if ( varPairSize != 2 )
          throw std::runtime_error("need two field name pairs for xfer");
        dofName = y_var[0].as<std::string>() ;
        functionName = y_var[1].as<std::string>() ;

        // push back pair of field names
        dofFunctionVec_.push_back(std::make_pair(dofName, functionName));
      }
    }
  }

  // deal with file name and banner
  if ( NaluEnv::self().parallel_rank() == 0 ) {
    std::ofstream myfile;
    myfile.open(outputFileName_.c_str());
    myfile << "Nalu Norm Post Processing......." << std::endl;
    myfile << "Field" << std::setw(w_) 
           << "Step" << std::setw(w_) << "Time" << std::setw(w_) << "Node Count" << std::setw(w_) 
           << "Loo"  << std::setw(w_) << "L1" << std::setw(w_)  << "L2" << std::setw(w_) << std::endl;
    myfile.close();
  }
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
SolutionNormPostProcessing::setup()
{
  // extract target names

  const std::vector<std::string> targetNames = realm_.get_physics_target_names();
  stk::mesh::MetaData & metaData = realm_.meta_data();

  // first, loop over all target names, extract the part and push back
  for ( size_t itarget = 0; itarget < targetNames.size(); ++itarget ) {
    stk::mesh::Part *targetPart = metaData.get_part(targetNames[itarget]);
    if ( NULL == targetPart ) {
      NaluEnv::self().naluOutputP0() << "Trouble with part " << targetNames[itarget] << std::endl;
      throw std::runtime_error("Sorry, no part name found by the name " + targetNames[itarget]);
    }
    else {
      // push back
      partVec_.push_back(targetPart);
    }
  }
  
  // iterate the vector
  sizeOfEachField_.resize(dofFunctionVec_.size());
  for ( size_t k = 0; k < dofFunctionVec_.size(); ++k ) {
        
    // extract the dof and function name
    const std::string dofName = dofFunctionVec_[k].first;
    const std::string functionName = dofFunctionVec_[k].second;
    
    // find the field
    const stk::mesh::FieldBase *dofField = metaData.get_field(stk::topology::NODE_RANK, dofName);
    if ( NULL == dofField )
      throw std::runtime_error("SolutionNorm::setup no dof field by the name of: " + dofName);
    
    // find the size; is there a better wat to determine a field size on a given part?
    const int dofSize = dofField->max_size(stk::topology::NODE_RANK);

    // increment total dof component size
    totalDofCompSize_ += dofSize;

    // save off size for each field
    sizeOfEachField_[k] = dofSize;

    // register the field, "dofName + _exact"
    const std::string dofNameExact = dofName + "_exact";
    
    stk::mesh::FieldBase *exactDofField
      = &(metaData.declare_field<stk::mesh::Field<double, stk::mesh::SimpleArrayTag> >(stk::topology::NODE_RANK, dofNameExact));
        
    // push back to vector of pairs; unique list 
    fieldPairVec_.push_back(std::make_pair(dofField, exactDofField));

    // loop over parts
    for ( size_t j = 0; j < partVec_.size(); ++j ) {
      // extract the part
      stk::mesh::Part *targetPart = partVec_[j];

      // put the field on the part
      stk::mesh::put_field_on_mesh(*exactDofField, *targetPart, dofSize, nullptr);
    
      // create the algorithm to populate the analytical field
      analytical_function_factory(functionName, exactDofField, targetPart);
    }
  } 
}

//--------------------------------------------------------------------------
//-------- analytical_function_factory -------------------------------------
//--------------------------------------------------------------------------
void
SolutionNormPostProcessing::analytical_function_factory(
  const std::string functionName,
  stk::mesh::FieldBase *exactDofField,
  stk::mesh::Part *part)
{
  AuxFunction *theAuxFunc = NULL;
  // switch on the name found...
  if ( functionName == "steady_2d_thermal" ) {
    theAuxFunc = new SteadyThermalContactAuxFunction();
  }
  else if ( functionName == "steady_3d_thermal" ) {
    theAuxFunc = new SteadyThermal3dContactAuxFunction();
  }
  else if ( functionName == "steady_3d_thermal_dtdx" ) {
    theAuxFunc = new SteadyThermal3dContactDtDxAuxFunction(0,realm_.meta_data().spatial_dimension());
  }
  else if ( functionName == "SteadyTaylorVortexVelocity" ) {
    theAuxFunc = new SteadyTaylorVortexVelocityAuxFunction(0,realm_.meta_data().spatial_dimension());
  }
  else if ( functionName == "VariableDensityVelocity" ) {
    theAuxFunc = new VariableDensityVelocityAuxFunction(0,realm_.meta_data().spatial_dimension());
  }
  else if ( functionName == "VariableDensityNonIsoVelocity" ) {
    theAuxFunc = new VariableDensityVelocityAuxFunction(0,realm_.meta_data().spatial_dimension());
  }
  else if ( functionName == "SteadyTaylorVortexGradPressure" ) {
    theAuxFunc = new SteadyTaylorVortexGradPressureAuxFunction(0,realm_.meta_data().spatial_dimension());
  }
  else if ( functionName == "VariableDensityMixtureFraction" ) {
    theAuxFunc = new VariableDensityMixFracAuxFunction();
  }
  else if ( functionName == "VariableDensityNonIsoTemperature" ) {
    theAuxFunc = new VariableDensityNonIsoTemperatureAuxFunction();
  }
  else if ( functionName == "BoussinesqNonIsoVelocity" ) {
    theAuxFunc = new BoussinesqNonIsoVelocityAuxFunction(0,realm_.meta_data().spatial_dimension());
  }
  else if ( functionName == "BoussinesqNonIsoTemperature" ) {
    theAuxFunc = new BoussinesqNonIsoTemperatureAuxFunction();
  }
  else if ( functionName == "kovasznay" ) {
    theAuxFunc = new KovasznayVelocityAuxFunction(0,realm_.meta_data().spatial_dimension());
  }
  else if ( functionName == "kovasznay_dpdx" ) {
    theAuxFunc = new KovasznayPressureGradientAuxFunction(0,realm_.meta_data().spatial_dimension());
  }
  else if ( functionName == "convecting_taylor_vortex" ) {
    theAuxFunc = new ConvectingTaylorVortexVelocityAuxFunction(0,realm_.meta_data().spatial_dimension());
  }
  else if ( functionName == "convecting_taylor_vortex_dpdx" ) {
    theAuxFunc = new ConvectingTaylorVortexPressureGradAuxFunction(0,realm_.meta_data().spatial_dimension());
  }
  else if ( functionName == "OneTwoTenVelocity" ) {
    theAuxFunc = new OneTwoTenVelocityAuxFunction(0,realm_.meta_data().spatial_dimension());
  }
  else if ( functionName == "wind_energy_taylor_vortex" ) {
    theAuxFunc = new WindEnergyTaylorVortexAuxFunction(0,realm_.meta_data().spatial_dimension(), std::vector<double>());
  }
  else if ( functionName == "wind_energy_taylor_vortex_dpdx" ) {
    theAuxFunc = new WindEnergyTaylorVortexPressureGradAuxFunction(0,realm_.meta_data().spatial_dimension(), std::vector<double>());
  }
  else {
    throw std::runtime_error(
      "SolutionNormPostProcessing::setup: Only "
      "steady_2d_thermal, steady_3d_thermal, steady_3d_thermal_dtdx, SteadyTaylorVortexVelocity, "
      "VariableDensityVelocity, VariableDensityNonIsoVelocity, SteadyTaylorVortexGradPressure, "
      "SteadyTaylorVortexGradPressure, VariableDensityNonIsoTemperature, kovasznay, "
      "kovasznay_dpdx, convecting_taylor_vortex, convecting_taylor_vortex_dpdx, "
      "wind_energy_taylor_vortex, wind_energy_taylor_vortex_dpdx, BoussinesqNonIso "
      "user functions supported");
  }

  // create the aux function
  AuxFunctionAlgorithm *auxAlg
    = new AuxFunctionAlgorithm(realm_, part,
                               exactDofField, theAuxFunc,
                               stk::topology::NODE_RANK);

  // push back
  populateExactNodalFieldAlg_.push_back(auxAlg);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
SolutionNormPostProcessing::execute()
{
  // check for proper count; return if not an output count (worry about field output sync?)
  const int timeStepCount = realm_.get_time_step_count();
  const bool processMe = (timeStepCount % outputFrequency_) == 0;
  if (!processMe)
    return;

  // determine norm  
  stk::mesh::MetaData &metaData = realm_.meta_data();
  stk::mesh::BulkData &bulkData = realm_.bulk_data();

  // populate the exact field
  for ( size_t k = 0; k < populateExactNodalFieldAlg_.size(); ++k )
    populateExactNodalFieldAlg_[k]->execute();
  
  stk::mesh::Selector s_locall_owned
    = metaData.locally_owned_part() 
    & stk::mesh::selectUnion(partVec_) 
    & !(realm_.get_inactive_selector());

  // size and initialize norm-related quantities; Loo, L1 and L2
  std::vector<double> l_LooNorm(totalDofCompSize_);
  std::vector<double> l_L12Norm(2*totalDofCompSize_);

  // initialize norms
  size_t l_nodeCount = 0;
  for ( int j = 0; j < totalDofCompSize_; ++j ) {
    l_LooNorm[j] = -1.0e16;
    l_L12Norm[j] = 0.0;
    l_L12Norm[totalDofCompSize_+j] = 0.0;
  }

  stk::mesh::BucketVector const& node_buckets = bulkData.get_buckets( stk::topology::NODE_RANK, s_locall_owned );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    l_nodeCount += length;
    
    int offSet = 0;
    for ( size_t j = 0; j < fieldPairVec_.size(); ++j ) {

      // extract fields
      const double *dofField = (double*)stk::mesh::field_data(*(fieldPairVec_[j].first), b);
      double *exactDofField = (double*)stk::mesh::field_data(*(fieldPairVec_[j].second), b);

      // size of this particular field      
      const int fieldSize = sizeOfEachField_[j];

      // initilize local counters
      double *Loo = &l_LooNorm[offSet];
      double *L1 = &l_L12Norm[offSet];
      double *L2 = &l_L12Norm[offSet+totalDofCompSize_];

      for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
        // loop over each field component
        for ( int i = 0; i < fieldSize; ++i ) {
          const double diff = std::abs(dofField[k*fieldSize+i] - exactDofField[k*fieldSize+i]);
          // norms...
          Loo[i] = std::max(diff, Loo[i]);
          L1[i] += diff;
          L2[i] += diff*diff;
        }
      }
      // increment offset
      offSet += fieldSize;
    }
  }

  // now assemble
  std::vector<double> g_LooNorm(totalDofCompSize_);
  std::vector<double> g_L12Norm(2*totalDofCompSize_);

  // initialize norms
  size_t g_nodeCount = 0;
  for ( int j = 0; j < totalDofCompSize_; ++j ) {
    g_LooNorm[j] = -1.0e16;
    g_L12Norm[j] = 0.0;
    g_L12Norm[totalDofCompSize_+j] = 0.0;
  }

  stk::ParallelMachine comm = NaluEnv::self().parallel_comm();
  stk::all_reduce_sum(comm, &l_nodeCount, &g_nodeCount, 1);
  stk::all_reduce_max(comm, &l_LooNorm[0], &g_LooNorm[0], totalDofCompSize_);
  stk::all_reduce_sum(comm, &l_L12Norm[0], &g_L12Norm[0], totalDofCompSize_*2);

  // output to a file
  if ( NaluEnv::self().parallel_rank() == 0 ) {
    const double currentTime = realm_.get_current_time();
    std::ofstream myfile;
    myfile.open(outputFileName_.c_str(), std::ios_base::app);

    int offSet = 0;
    for ( size_t j = 0; j < fieldPairVec_.size(); ++j ) {
      const stk::mesh::FieldBase *dofField = fieldPairVec_[j].first;
      const int fieldSize = sizeOfEachField_[j];
      const std::string dofName = dofField->name();
      for ( int i = 0; i < fieldSize; ++i ) {
        myfile << std::setprecision(percision_) 
               << std::setw(w_) 
               << dofName  << "[" << i << "]" << std::setw(w_)
               << timeStepCount << std::setw(w_)
               << currentTime << std::setw(w_) 
               << g_nodeCount << std::setw(w_) 
               << g_LooNorm[offSet+i] << std::setw(w_)
               << g_L12Norm[offSet+i]/g_nodeCount << std::setw(w_)
               << std::sqrt(g_L12Norm[offSet+i+totalDofCompSize_]/g_nodeCount) << std::setw(w_)
               << std::endl;
      }
      // increment offset
      offSet += fieldSize;
    }
  }
}

} // namespace nalu
} // namespace Sierra
