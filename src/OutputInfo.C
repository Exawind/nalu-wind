// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <OutputInfo.h>
#include <NaluEnv.h>
#include <NaluParsing.h>

// ioss
#include <Ioss_PropertyManager.h>
#include <Ioss_Property.h>

// basic c++
#include <stdexcept>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// OutputInfo - holder for output information held at realm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
OutputInfo::OutputInfo() 
  : outputDBName_("output.e"),
    catalystFileName_(""),
    catalystParseJson_(""),
    paraviewScriptName_(""),
    outputFreq_(1),
    outputStart_(0),
    outputNodeSet_(false),
    serializedIOGroupSize_(0),
    hasOutputBlock_(false),
    hasRestartBlock_(false),
    activateRestart_(false),
    meshAdapted_(false),
    restartTime_(0.0),
    restartDBName_("restart.rst"),
    restartFreq_(500),
    restartStart_(0),
    restartMaxDataBaseStepSize_(100000),
    restartNodeSet_(true),
    outputCompressionLevel_(0),
    outputCompressionShuffle_(false),
    restartCompressionLevel_(0),
    restartCompressionShuffle_(false),
    userWallTimeResults_(false, 1.0e6),
    userWallTimeRestart_(false, 1.0e6),
    outputPropertyManager_(new Ioss::PropertyManager()),
    restartPropertyManager_(new Ioss::PropertyManager())  
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
OutputInfo::~OutputInfo()
{
  delete outputPropertyManager_;
  delete restartPropertyManager_;
}

//--------------------------------------------------------------------------
//-------- load ------------------------------------------------------------
//--------------------------------------------------------------------------
void
OutputInfo::load(
  const YAML::Node & y_node)
{

  const YAML::Node y_output = y_node["output"];
  if (y_output)
  {
    // user desires output
    hasOutputBlock_ = true;

    // output data base name
    get_if_present(y_output, "output_data_base_name", outputDBName_, outputDBName_);

    // catalyst file name
    get_if_present(y_output, "catalyst_file_name", catalystFileName_, catalystFileName_);

    // paraview script name
    get_if_present(y_output, "paraview_script_name", paraviewScriptName_, paraviewScriptName_);
    
    // output frequency
    get_if_present(y_output, "output_frequency", outputFreq_, outputFreq_);

    // output start
    get_if_present(y_output, "output_start", outputStart_, outputStart_);

    // output at WALL time
    if ( y_output["output_forced_wall_time"]) {
      userWallTimeResults_.first = true;
      userWallTimeResults_.second = y_output["output_forced_wall_time"].as<double>() ;
    }

    // determine if we want nodeset output
    get_if_present(y_output, "output_node_set", outputNodeSet_, outputNodeSet_);
    
    // compression options; add to manager
    if ( y_output["compression_level"] ) {
      outputCompressionLevel_ = y_output["compression_level"].as<int>() ;
      outputPropertyManager_->add(Ioss::Property("COMPRESSION_LEVEL", outputCompressionLevel_));
      
      // when compression is active, add netcdf4 file type
      outputPropertyManager_->add(Ioss::Property("FILE_TYPE", "netcdf4"));
      
      // only allow for shuffle if compression is active
      get_if_present(y_output, "compression_shuffle", outputCompressionShuffle_, outputCompressionShuffle_);
      if ( outputCompressionShuffle_ ) {
        const int cs = 1;
        outputPropertyManager_->add(Ioss::Property("COMPRESSION_SHUFFLE", cs));
      }
    }
    // error checking
    if ( outputCompressionShuffle_ )
      if ( outputCompressionLevel_ == 0 ) 
        NaluEnv::self().naluOutputP0() << "OutputInfo::load() Output Warning: One should not shuffle if one is not compressing" << std::endl;
    
    // serialize io...
    {
      get_if_present(y_output, "serialized_io_group_size", serializedIOGroupSize_, serializedIOGroupSize_);
      if (serializedIOGroupSize_) {
        NaluEnv::self().naluOutputP0() << "Info: found non-zero serialized_io_group_size in input file= " << serializedIOGroupSize_ << std::endl;
      }
    }

    const YAML::Node y_vars = y_output["output_variables"];
    if (y_vars)
    {
      size_t varSize = y_vars.size();
      for (size_t ioption = 0; ioption < varSize; ++ioption)
      {
        const YAML::Node y_var = y_vars[ioption];
        std::string fieldName;
        fieldName = y_var.as<std::string>() ;
        outputFieldNameSet_.insert(fieldName);
      }
    }
  }
  
  // output for restart
  const YAML::Node y_restart = y_node["restart"];
  if (y_restart)
  {    
    // some sort of intent to manage a restart event - either clean or restart run
    hasRestartBlock_ = true;

    // restart output data base name
    get_if_present(y_restart, "restart_data_base_name", restartDBName_, restartDBName_);
    
    // restart output frequency
    get_if_present(y_restart, "restart_frequency", restartFreq_, restartFreq_);
    
    // restart start
    get_if_present(y_restart, "restart_start", restartStart_, restartStart_);

    // output at WALL time
    if ( y_restart["restart_forced_wall_time"]) {
      userWallTimeRestart_.first = true;
      userWallTimeRestart_.second = y_restart["restart_forced_wall_time"].as<double>() ;
    }

    // determine if we want nodeset restart output
    get_if_present(y_restart, "restart_node_set", restartNodeSet_, restartNodeSet_);
    
    // max data base size for restart
    get_if_present(y_restart, "max_data_base_step_size", restartMaxDataBaseStepSize_, restartMaxDataBaseStepSize_);
    
    // compression options; add to manager
    if ( y_restart["compression_level"] ) {
      restartCompressionLevel_ = y_restart["compression_level"].as<int>() ;
      restartPropertyManager_->add(Ioss::Property("COMPRESSION_LEVEL", restartCompressionLevel_));
      
      // when compression is active, add netcdf4 file type
      restartPropertyManager_->add(Ioss::Property("FILE_TYPE", "netcdf4"));
      
      // only allow for shuffle if compression is active
      get_if_present(y_restart, "compression_shuffle", restartCompressionShuffle_, restartCompressionShuffle_);
      if ( restartCompressionShuffle_ ) {
        const int cs = 1;
        restartPropertyManager_->add(Ioss::Property("COMPRESSION_SHUFFLE", cs));
      }
    }
    // error checking
    if ( restartCompressionShuffle_ )
      if ( restartCompressionLevel_ == 0 )  
        NaluEnv::self().naluOutputP0() << "OutputInfo::load() Restart Warning: One should not shuffle if one is not compressing" << std::endl;
    
    // check to see if restart is active for this run
    if ( y_restart["restart_time"] ) {
      activateRestart_ = true;
      restartTime_ = y_restart["restart_time"].as<double>() ;
    }
    
    const YAML::Node y_vars = y_restart["restart_variables"];
    if (y_vars) {
      NaluEnv::self().naluOutputP0() << "Restart variable specification has been deprecated" << std::endl;
    }
  }
}

int
OutputInfo::get_restart_frequency() {
  return restartFreq_;
}
// compression options
int
OutputInfo::get_output_compression() {
  return outputCompressionLevel_;
}
bool
OutputInfo::get_output_shuffle() {
  return outputCompressionShuffle_;
}

int
OutputInfo::get_restart_compression() {
  return restartCompressionLevel_;
}

bool
OutputInfo::get_restart_shuffle() {
  return restartCompressionShuffle_;
}

} // namespace nalu
} // namespace Sierra
