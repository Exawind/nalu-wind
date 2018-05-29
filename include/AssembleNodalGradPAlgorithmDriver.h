/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef AssembleNodalGradPAlgorithmDriver_h
#define AssembleNodalGradPAlgorithmDriver_h

#include<AlgorithmDriver.h>
#include <FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class AssembleNodalGradPAlgorithmDriver : public AlgorithmDriver
{
public:

  AssembleNodalGradPAlgorithmDriver(
    Realm &realm);
  virtual ~AssembleNodalGradPAlgorithmDriver() {}

  virtual void pre_work() override;
  virtual void post_work() override;

};

} // namespace nalu
} // namespace Sierra

#endif
