// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef NODEKERNELUTILS_H
#define NODEKERNELUTILS_H

#include "AssembleNGPNodeSolverAlgorithm.h"
#include "Enums.h"
#include "EquationSystem.h"
#include "Realm.h"
#include "SolutionOptions.h"

#include <map>

namespace sierra {
namespace nalu {

template<typename LambdaGeneral, typename LambdaUserSrc>
void process_ngp_node_kernels(
  std::map<AlgorithmType, SolverAlgorithm*>& solverAlgMap,
  Realm& realm,
  stk::mesh::Part* part,
  EquationSystem* eqSystem,
  LambdaGeneral lambdaGeneral,
  LambdaUserSrc lambdaUsrSrc)
{
  const auto algMass = AlgorithmType::MASS;
  const auto it = solverAlgMap.find(algMass);

  if (it == solverAlgMap.end()) {
    auto* nodeAlg = new AssembleNGPNodeSolverAlgorithm(realm, part, eqSystem);
    solverAlgMap[algMass] = nodeAlg;

    // Custom node src kernels for this equation system
    lambdaGeneral(*nodeAlg);

    const auto it = realm.solutionOptions_->srcTermsMap_.find(eqSystem->eqnTypeName_);
    if (it != realm.solutionOptions_->srcTermsMap_.end()) {
      NaluEnv::self().naluOutputP0()
        << "Processing user source terms for "
        << eqSystem->eqnTypeName_ << std::endl;
      for (auto& srcName : it->second) {
        lambdaUsrSrc(*nodeAlg, srcName);
      }
    }
  } else {
    it->second->partVec_.push_back(part);
  }
}

}  // nalu
}  // sierra

#endif /* NODEKERNELUTILS_H */
