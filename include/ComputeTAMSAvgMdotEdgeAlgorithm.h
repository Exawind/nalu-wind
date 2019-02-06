/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ComputeTAMSAvgMdotEdgeAlgorithm_h
#define ComputeTAMSAvgMdotEdgeAlgorithm_h

#include <Algorithm.h>
#include <FieldTypeDef.h>

// stk
#include <stk_mesh/base/Part.hpp>

namespace sierra {
namespace nalu {

class Realm;

class ComputeTAMSAvgMdotEdgeAlgorithm : public Algorithm
{
public:
  ComputeTAMSAvgMdotEdgeAlgorithm(Realm& realm, stk::mesh::Part* part);
  ~ComputeTAMSAvgMdotEdgeAlgorithm();

  void execute();

  const bool meshMotion_;
  VectorFieldType* velocityRTM_;
  VectorFieldType* Gpdx_;
  VectorFieldType* coordinates_;
  ScalarFieldType* pressure_;
  ScalarFieldType* avgTime_;
  ScalarFieldType* density_;
  VectorFieldType* edgeAreaVec_;
  ScalarFieldType* massFlowRate_;
  ScalarFieldType* avgMassFlowRate_;
};

} // namespace nalu
} // namespace sierra

#endif
