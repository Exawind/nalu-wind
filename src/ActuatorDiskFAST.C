/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
/*
 * ActuatorDiskFAST.C
 *
 *  Created on: Oct 16, 2018
 *      Author: psakiev
 */

#ifdef NALU_USES_OPENFAST
#include "ActuatorDiskFAST.h"

namespace sierra{
namespace nalu{

  void ActuatorDiskFAST::update_class_specific(){
    Actuator::complete_search();
  }

  void ActuatorDiskFAST::create_point_info_map_class_specific(){
    if( !FAST.isDryRun() ){
      for (size_t iTurb = 0; iTurb < actuatorInfo_.size(); ++iTurb){
         const auto actuatorDiskInfo = dynamic_cast<ActuatorDiskFASTInfo*>(actuatorInfo_[iTurb].get());
         if(actuatorDiskInfo == NULL){
           throw std::runtime_error("Object in ActuatorInfo is not the correct type. It should be ActuatorDiskFASTInfo.");
         }

         int processorID = FAST.get_procNo(iTurb);

         const int numFastForcePts = FAST.get_numForcePts(iTurb);
         const int numTotalForcePts = numFastForcePts + FAST.get_numBlades(iTurb);



       }
    }
  }

  std::string ActuatorDiskFAST::get_class_name(){
    return "ActuatorDiskFAST";
  }

}
}
#endif
