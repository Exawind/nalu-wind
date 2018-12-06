/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef COMPUTEMETRICTENSORELEMALGORITHM_H
#define COMPUTEMETRICTENSORELEMALGORITHM_H

#include <Algorithm.h>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;
class ComputeMetricTensorElemAlgorithm : public Algorithm {
public:
  ComputeMetricTensorElemAlgorithm(Realm &realm, stk::mesh::Part *part);
  virtual ~ComputeMetricTensorElemAlgorithm() {}

  virtual void execute();

  VectorFieldType *coordinates_{nullptr};
  GenericFieldType *Mij_{nullptr};

  std::vector<double> ws_coordinates;
  std::vector<double> ws_dndx;
  std::vector<double> ws_deriv;
  std::vector<double> ws_det_j;
  std::vector<double> ws_Mij;
};

} // namespace nalu
} // namespace sierra

#endif
