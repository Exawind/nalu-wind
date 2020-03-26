// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef SolutionNormPostProcessing_h
#define SolutionNormPostProcessing_h

#include <NaluParsedTypes.h>

#include <string>
#include <vector>
#include <utility>

// stk forwards
namespace stk {
  namespace mesh {
    class BulkData;
    class FieldBase;
    class MetaData;
    class Part;
    typedef std::vector<Part*> PartVector;
  }
}

namespace sierra{
namespace nalu{

class AuxFunctionAlgorithm;
class Realm;

class SolutionNormPostProcessing
{
public:
  
  SolutionNormPostProcessing(
    Realm &realm,
    const YAML::Node &node);
  ~SolutionNormPostProcessing();
  
  // load all of the options
  void load(
    const YAML::Node & node);

  // setup nodal field registration; algorithms
  void setup();

  // create the algorithm to populate the exact field
  void analytical_function_factory(
    const std::string functionName,
    stk::mesh::FieldBase *exactDofField,
    stk::mesh::Part *part);

  // populate nodal field and output norms (if appropriate)
  void execute();

  // hold the realm
  Realm &realm_;

  // how often to output norms
  int outputFrequency_;
  
  // keep track of total dof component size
  int totalDofCompSize_;

  // for each dof, save off field size
  std::vector<int> sizeOfEachField_;
  
  // file name
  std::string outputFileName_;
  
  // spacing
  int w_;

  // percision
  int percision_;

  // hold the dofName, functionName in a vector 
  std::vector<std::pair<std::string, std::string> > dofFunctionVec_;

  // hold the dofField and exactDofField
  std::vector<std::pair<const stk::mesh::FieldBase*, stk::mesh::FieldBase*> > fieldPairVec_;

  // vector of parts for post processing
  stk::mesh::PartVector partVec_;

  // vector of algorithms that process the analytical field
  std::vector<AuxFunctionAlgorithm *> populateExactNodalFieldAlg_;
};

} // namespace nalu
} // namespace Sierra

#endif
