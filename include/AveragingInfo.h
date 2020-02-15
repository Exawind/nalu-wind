// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef AveragingInfo_h
#define AveragingInfo_h

#include <string>
#include <vector>

namespace stk {
  namespace mesh {
    class FieldBase;
    class Part;
    typedef std::vector<Part*> PartVector;
  }
}

namespace sierra{
namespace nalu{

class AveragingInfo
{
public:

  AveragingInfo();
  ~AveragingInfo();

  // name of this block
  std::string name_;

  // specialty options
  bool computeReynoldsStress_;
  bool computeTke_;
  bool computeFavreStress_;
  bool computeFavreTke_;
  bool computeResolvedStress_{false};
  bool computeSFSStress_{false};
  bool computeVorticity_;
  bool computeQcriterion_;
  bool computeLambdaCI_;
  bool computeMeanResolvedKe_;

  // Temperature stresses
  bool computeTemperatureSFS_{false};
  bool computeTemperatureResolved_{false};
  
  // vector of part names, e.g., block_1, surface_2
  std::vector<std::string> targetNames_;

  // vector of parts
  stk::mesh::PartVector partVec_;

  // vector of favre/reynolds fields
  std::vector<std::string> favreFieldNameVec_;
  std::vector<std::string> reynoldsFieldNameVec_;
  std::vector<std::string> resolvedFieldNameVec_;
  std::vector<std::string> movingAvgFieldNameVec_;


  // vector of pairs of fields
  std::vector<std::pair<stk::mesh::FieldBase *, stk::mesh::FieldBase *> > favreFieldVecPair_;
  std::vector<std::pair<stk::mesh::FieldBase *, stk::mesh::FieldBase *> > reynoldsFieldVecPair_;
  std::vector<std::pair<stk::mesh::FieldBase *, stk::mesh::FieldBase *> > resolvedFieldVecPair_;

  // sizes for each
  std::vector<unsigned> favreFieldSizeVec_;
  std::vector<unsigned> reynoldsFieldSizeVec_;
  std::vector<unsigned> resolvedFieldSizeVec_;
};

} // namespace nalu
} // namespace Sierra

#endif
