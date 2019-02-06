/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ComputeTAMSAvgMdotElemAlgorithm_h
#define ComputeTAMSAvgMdotElemAlgorithm_h

#include <Algorithm.h>
#include <FieldTypeDef.h>

// stk
#include <stk_mesh/base/Part.hpp>

namespace sierra {
namespace nalu {

class Realm;

class ComputeTAMSAvgMdotElemAlgorithm : public Algorithm
{
public:
  ComputeTAMSAvgMdotElemAlgorithm(Realm& realm, stk::mesh::Part* part);
  ~ComputeTAMSAvgMdotElemAlgorithm();

  void execute();

  // extract fields; nodal
  VectorFieldType* velocityRTM_;
  VectorFieldType* coordinates_;
  ScalarFieldType* density_;
  ScalarFieldType* avgTime_;
  GenericFieldType* massFlowRate_;
  GenericFieldType* avgMassFlowRate_;

  const bool shiftTAMSAvgMdot_;
};

} // namespace nalu
} // namespace sierra

#endif
