// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef SolverAlgorithmDriver_h
#define SolverAlgorithmDriver_h

#include<AlgorithmDriver.h>
#include<Enums.h>

#include<map>

namespace sierra{
namespace nalu{

class Realm;
class SolverAlgorithm;

class SolverAlgorithmDriver : public AlgorithmDriver
{
public:

  SolverAlgorithmDriver(
    Realm &realm);
  virtual ~SolverAlgorithmDriver();

  virtual void initialize_connectivity();
  virtual void pre_work();
  virtual void execute();
  virtual void post_work();
  
  // different types of algorithms... interior/flux; constraints and dirichlet
  std::map<std::string, SolverAlgorithm *> solverAlgorithmMap_;
  std::map<AlgorithmType, SolverAlgorithm *> solverAlgMap_;
  std::map<AlgorithmType, SolverAlgorithm *> solverConstraintAlgMap_;
  std::map<AlgorithmType, SolverAlgorithm *> solverDirichAlgMap_;
};

} // namespace nalu
} // namespace Sierra

#endif
