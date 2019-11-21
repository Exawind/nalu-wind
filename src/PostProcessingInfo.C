// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <PostProcessingInfo.h>
#include <PostProcessingData.h>
#include <NaluParsing.h>

// basic c++
#include <stdexcept>

namespace sierra{
namespace nalu{


//==========================================================================
// Class Definition
//==========================================================================
// PostProcessingInfo - holder for post processing information held at realm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
PostProcessingInfo::PostProcessingInfo() 
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
PostProcessingInfo::~PostProcessingInfo()
{
  std::vector<PostProcessingData *>::iterator ii;
  for( ii=ppDataVec_.begin(); ii!=ppDataVec_.end(); ++ii )
    delete *ii;
}
  
//--------------------------------------------------------------------------
//-------- load ------------------------------------------------------------
//--------------------------------------------------------------------------
void
PostProcessingInfo::load(
    const YAML::Node & y_node)
{

  // post processing on?
  const bool optional = true;
  const YAML::Node y_pp = expect_sequence(y_node, "post_processing", optional);
  if (y_pp) {

    for (size_t itype = 0; itype < y_pp.size(); ++itype) {

      // extract the particular type
      const YAML::Node y_type = y_pp[itype] ;
  
      // create the data
      PostProcessingData *ppData = new PostProcessingData();      

      // push the data to vector
      ppDataVec_.push_back(ppData);

      // save the type; surface/node/volume
      ppData->type_ = y_type["type"].as<std::string>() ;
      
      // physics; traction, etc.
      if ( y_type["physics"] )
        ppData->physics_ = y_type["physics"].as<std::string>() ;
      else
        throw std::runtime_error("parser error PostProcessings::load: no physics type specified");
      
      // outfile file
      if ( y_type["output_file_name"] )
        ppData->outputFileName_ = y_type["output_file_name"].as<std::string>() ;
      else
        throw std::runtime_error("parser error PostProcessings::load: no output file specified");
      
      // frequency
      if ( y_type["frequency"])
        ppData->frequency_ = y_type["frequency"].as<int>() ;

      // misc parameters
      if ( y_type["parameters"] ) {

        // extract the value(s)
        const YAML::Node targets = y_type["parameters"];
        if (targets.Type() == YAML::NodeType::Scalar) {
          ppData->parameters_.resize(1);
          ppData->parameters_[0] = targets.as<double>() ;
        }
        else {
          ppData->parameters_.resize(targets.size());
          for (size_t i=0; i < targets.size(); ++i) {
            ppData->parameters_[i] = targets[i].as<double>() ;
          }
        }
      }
      
      // extract the target(s)
      const YAML::Node targets = y_type["target_name"];
      if (targets.Type() == YAML::NodeType::Scalar) {
        ppData->targetNames_.resize(1);
        ppData->targetNames_[0] = targets.as<std::string>() ;
      }
      else {
        ppData->targetNames_.resize(targets.size());
	for (size_t i=0; i < targets.size(); ++i) {
          ppData->targetNames_[i] = targets[i].as<std::string>() ;
        }
      }
    }
  }
 
}


} // namespace nalu
} // namespace Sierra
