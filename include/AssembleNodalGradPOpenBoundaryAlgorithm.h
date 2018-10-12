/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef AssembleNodalGradPOpenBoundaryAlgorithm_h
#define AssembleNodalGradPOpenBoundaryAlgorithm_h

#include<Algorithm.h>
#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class AssembleNodalGradPOpenBoundaryAlgorithm : public Algorithm
{
public:
  AssembleNodalGradPOpenBoundaryAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    const bool useShifted);
  virtual ~AssembleNodalGradPOpenBoundaryAlgorithm() {}

  virtual void execute();

  const bool useShifted_;
  const bool zeroGrad_;
  const bool massCorr_;
};

} // namespace nalu
} // namespace Sierra

#endif
