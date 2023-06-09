// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef Algorithm_h
#define Algorithm_h

#include <vector>

namespace stk {
namespace mesh {
class Part;
typedef std::vector<Part*> PartVector;
} // namespace mesh
} // namespace stk
namespace sierra {
namespace nalu {

class Realm;
class MasterElement;
class SupplementalAlgorithm;
class Kernel;

class Algorithm
{
public:
  // provide part
  Algorithm(Realm& realm, stk::mesh::Part* part);

  // provide part vector
  Algorithm(Realm& realm, const stk::mesh::PartVector& partVec);

  virtual ~Algorithm();

  virtual void execute() = 0;

  virtual void pre_work() {}

  Realm& realm_;
  stk::mesh::PartVector partVec_;
  std::vector<SupplementalAlgorithm*> supplementalAlg_;

  std::vector<Kernel*> activeKernels_;
};

} // namespace nalu
} // namespace sierra

#endif
