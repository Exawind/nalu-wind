// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#pragma once

#include "ngp_algorithms/FluxDivAlgDriver.h"
#include "ngp_algorithms/FluxDivBndryElemAlg.h"
#include "ngp_algorithms/FluxDivEdgeAlg.h"
#include "FieldTypeDef.h"

namespace sierra::kynema_ugf {

class ContinuityResidualAlgDriver : public NgpAlgDriver
{
public:
  ContinuityResidualAlgDriver(Realm&);
  void pre_work() final;
  void execute() final;
  void post_work() final;

  template <typename EdgeAlg, class... Args>
  void register_edge_algorithm(
    AlgorithmType algType,
    stk::mesh::Part* part,
    const std::string& algSuffix,
    Args&&... args)
  {
    div_mdot_algs_.register_edge_algorithm<EdgeAlg>(
      algType, part, algSuffix, std::forward<Args>(args)...);
  }

  template <template <typename> class FaceAlg, class... Args>
  void register_face_algorithm(
    AlgorithmType algType,
    stk::mesh::Part* part,
    const std::string& algSuffix,
    Args&&... args)
  {
    div_mdot_algs_.register_face_algorithm<FaceAlg>(
      algType, part, algSuffix, std::forward<Args>(args)...);
  }

  template <template <typename> class ElemAlg, class... Args>
  void register_elem_algorithm(
    AlgorithmType /*algType*/,
    stk::mesh::Part* /*part*/,
    const std::string& /*algSuffix*/,
    Args&&... /*args*/)
  {
    STK_ThrowRequire(false);
  }

  template <typename Algorithm, class... Args>
  void register_legacy_algorithm(
    AlgorithmType /*algType*/,
    stk::mesh::Part* /*part*/,
    const std::string& /*algSuffix*/,
    Args&&... /*args*/)
  {
    STK_ThrowRequire(false);
  }

  template <template <typename> class FaceElemAlg, class... Args>
  void register_face_elem_algorithm(
    AlgorithmType /*algType*/,
    stk::mesh::Part* /*part*/,
    const stk::topology /*elemTopo*/,
    const std::string& /*algSuffix*/,
    Args&&... /*args*/)
  {
    STK_ThrowRequire(false);
  }

private:
  FluxDivAlgDriver div_mdot_algs_;
};

} // namespace sierra::kynema_ugf
