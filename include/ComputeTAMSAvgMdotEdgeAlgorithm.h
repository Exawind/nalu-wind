/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
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
  virtual ~ComputeTAMSAvgMdotEdgeAlgorithm() = default;

  void execute();

  const bool meshMotion_;
  ScalarFieldType* avgTime_;
  ScalarFieldType* massFlowRate_;
  ScalarFieldType* avgMassFlowRate_;
};

} // namespace nalu
} // namespace sierra

#endif
