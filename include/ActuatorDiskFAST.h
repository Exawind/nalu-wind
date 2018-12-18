/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
/*
 * ActuatorDiskFAST.h
 *
 *  Created on: Oct 16, 2018
 *      Author: psakiev
 */

#ifndef ACTUATORDISKFAST_H_
#define ACTUATORDISKFAST_H_

#include "ActuatorFAST.h"

/*
 * ideas:
 * ==================================================================================
 * 1) have a vector of points associated with each forcing point
 * -------------------------------------------------------------
 *    - pros:
 *        * no change to api regarding fast
 *    - cons:
 *        * set up search, ghosting and interpolation to work with this format
 *        * set up spreading to work with this format
 *
 * -------------------------------------------------------------
 * 2) Put disk points into normal point map
 * -------------------------------------------------------------
 *    - pros:
 *        * execution of force spreading is the same
 *        * ghosting, search same
 *    - cons:
 *        * have to figure out a way to only send a portion of the points to fast
 */

namespace sierra{
namespace nalu{

/** Class to hold additional information for the disk points that
 *  that will need to be populated from the actuator line sampling
 *  in FAST
 *
 */
class ActuatorDiskFASTInfo : public ActuatorFASTInfo{
public:
  ActuatorDiskFASTInfo();
  ~ActuatorDiskFASTInfo();
  int numberOfPointsBetweenLines_;
};

class ActuatorDiskFAST : public ActuatorFAST{

  void create_point_info_map_class_specific() override;

  void update_class_specific() override;

  void load_class_specific() override;

  void execute_class_specific() override;

  std::string get_class_name() override;
};
} // namespace nalu
} // namespace sierra
#endif /* ACTUATORDISKFAST_H_ */
