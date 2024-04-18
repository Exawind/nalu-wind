// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef KernelBuilder_h
#define KernelBuilder_h

#include <kernel/Kernel.h>
#include <AssembleElemSolverAlgorithm.h>
#include <AssembleFaceElemSolverAlgorithm.h>
#include <EquationSystem.h>
#include <AlgTraits.h>
#include <NaluEnv.h>
#include <kernel/KernelBuilderLog.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_topology/topology.hpp>

#include <BuildTemplates.h>

#include <algorithm>
#include <tuple>

namespace sierra {
namespace nalu {
class Realm;

inline std::pair<AssembleElemSolverAlgorithm*, bool>
build_or_add_part_to_solver_alg(
  EquationSystem& eqSys,
  stk::mesh::Part& part,
  std::map<std::string, SolverAlgorithm*>& solverAlgs)
{
  const stk::topology topo = part.topology();
  const std::string algName =
    eqSys.name_ + "_AssembleElemSolverAlg_" + topo.name();

  bool isNotNGP =
    !(topo == stk::topology::HEXAHEDRON_8 ||
      topo == stk::topology::HEXAHEDRON_27 ||
      topo == stk::topology::QUADRILATERAL_4_2D ||
      topo == stk::topology::TRIANGLE_3_2D || topo == stk::topology::WEDGE_6 ||
      topo == stk::topology::TETRAHEDRON_4 || topo == stk::topology::PYRAMID_5);
  STK_ThrowRequireMsg(
    !isNotNGP, "Consolidated algorithm called on non-NGP MasterElement");

  auto itc = solverAlgs.find(algName);
  bool createNewAlg = itc == solverAlgs.end();
  if (createNewAlg) {
    auto* theSolverAlg = new AssembleElemSolverAlgorithm(
      eqSys.realm_, &part, &eqSys, stk::topology::ELEMENT_RANK,
      topo.num_nodes());
    STK_ThrowRequire(theSolverAlg != nullptr);

    NaluEnv::self().naluOutputP0()
      << "Created the following interior elem alg: " << algName << std::endl;
    solverAlgs.insert({algName, theSolverAlg});
  } else {
    auto& partVec = itc->second->partVec_;
    if (std::find(partVec.begin(), partVec.end(), &part) == partVec.end()) {
      partVec.push_back(&part);
    }
  }

  auto* theSolverAlg =
    dynamic_cast<AssembleElemSolverAlgorithm*>(solverAlgs.at(algName));
  STK_ThrowRequire(theSolverAlg != nullptr);

  return {theSolverAlg, createNewAlg};
}

template <template <typename> class T, typename... Args>
Kernel*
build_fem_kernel(stk::topology topo, Args&&... args)
{
  STK_ThrowRequireMsg(
    topo == stk::topology::HEXAHEDRON_8,
    "FEM kernels only implemented for Hex8 topology");
  return new T<AlgTraitsHex8>(std::forward<Args>(args)...);
}

template <template <typename> class T, typename... Args>
Kernel*
build_topo_kernel(stk::topology topo, Args&&... args)
{
  switch (topo.value()) {
  case stk::topology::HEX_8:
    return new T<AlgTraitsHex8>(std::forward<Args>(args)...);
  case stk::topology::TET_4:
    return new T<AlgTraitsTet4>(std::forward<Args>(args)...);
  case stk::topology::PYRAMID_5:
    return new T<AlgTraitsPyr5>(std::forward<Args>(args)...);
  case stk::topology::WEDGE_6:
    return new T<AlgTraitsWed6>(std::forward<Args>(args)...);
  case stk::topology::QUAD_4_2D:
    return new T<AlgTraitsQuad4_2D>(std::forward<Args>(args)...);
  case stk::topology::TRI_3_2D:
    return new T<AlgTraitsTri3_2D>(std::forward<Args>(args)...);
  default:
    return nullptr;
  }
}

class KernelBuilder
{
public:
  KernelBuilder(
    EquationSystem& eqSys,
    stk::mesh::Part& part,
    std::map<std::string, SolverAlgorithm*>& solverAlgs)
    : eqSys_(eqSys), part_(part)
  {
    std::tie(solverAlg_, solverAlgWasBuilt_) =
      build_or_add_part_to_solver_alg(eqSys, part, solverAlgs);
  }

  void report()
  {
    if (solverAlgWasBuilt_) {
      eqSys_.report_built_supp_alg_names();
      eqSys_.report_invalid_supp_alg_names();
    }
  }

  ElemDataRequests& data_prereqs() { return solverAlg_->dataNeededByKernels_; }

  template <template <typename> class T, typename... Args>
  bool build_topo_kernel_if_requested(std::string name, Args&&... args)
  {
    if (solverAlgWasBuilt_) {
      bool isCreated = false;
      KernelBuilderLog::self().add_valid_name(eqSys_.eqnTypeName_, name);
      if (eqSys_.supp_alg_is_requested(name)) {
        Kernel* compKernel =
          build_topo_kernel<T>(part_.topology(), std::forward<Args>(args)...);
        STK_ThrowRequire(compKernel != nullptr);
        KernelBuilderLog::self().add_built_name(eqSys_.eqnTypeName_, name);
        solverAlg_->activeKernels_.push_back(compKernel);
        isCreated = true;
      }
      return isCreated;
    }
    return false;
  }

  template <template <typename> class T, typename... Args>
  bool build_fem_kernel_if_requested(std::string name, Args&&... args)
  {
    if (solverAlgWasBuilt_) {
      bool isCreated = false;
      KernelBuilderLog::self().add_valid_name(eqSys_.eqnTypeName_, name);
      if (eqSys_.supp_alg_is_requested(name)) {
        Kernel* compKernel =
          build_fem_kernel<T>(part_.topology(), std::forward<Args>(args)...);
        STK_ThrowRequire(compKernel != nullptr);
        KernelBuilderLog::self().add_built_name(eqSys_.eqnTypeName_, name);
        solverAlg_->activeKernels_.push_back(compKernel);
        isCreated = true;
      }
      return isCreated;
    }
    return false;
  }

private:
  EquationSystem& eqSys_;
  stk::mesh::Part& part_;
  AssembleElemSolverAlgorithm* solverAlg_{nullptr};
  bool solverAlgWasBuilt_{false};
};

template <template <typename> class T, typename... Args>
Kernel*
build_face_elem_topo_kernel(
  stk::topology faceTopo, stk::topology elemTopo, Args&&... args)
{
  switch (faceTopo.value()) {
  case stk::topology::QUAD_4:
    switch (elemTopo) {
    case stk::topology::HEX_8:
      return new T<AlgTraitsQuad4Hex8>(std::forward<Args>(args)...);
    case stk::topology::PYRAMID_5:
      return new T<AlgTraitsQuad4Pyr5>(std::forward<Args>(args)...);
    case stk::topology::WEDGE_6:
      return new T<AlgTraitsQuad4Wed6>(std::forward<Args>(args)...);
    default:
      STK_ThrowRequireMsg(
        false, "Quad4 exposed face is not attached to either a hex8, pyr5, or "
               "wedge6.");
      return nullptr;
    }
  case stk::topology::TRI_3:
    switch (elemTopo) {
    case stk::topology::TET_4:
      return new T<AlgTraitsTri3Tet4>(std::forward<Args>(args)...);
    case stk::topology::PYRAMID_5:
      return new T<AlgTraitsTri3Pyr5>(std::forward<Args>(args)...);
    case stk::topology::WEDGE_6:
      return new T<AlgTraitsTri3Wed6>(std::forward<Args>(args)...);
    default:
      STK_ThrowRequireMsg(
        false,
        "Tri3 exposed face is not attached to either a tet4, pyr5, or wedge6.");
      return nullptr;
    }
  case stk::topology::LINE_2:
    switch (elemTopo) {
    case stk::topology::TRI_3_2D:
      return new T<AlgTraitsEdge2DTri32D>(std::forward<Args>(args)...);
    default:
      return new T<AlgTraitsEdge2DQuad42D>(std::forward<Args>(args)...);
    }
  default:
    return nullptr;
  }
}

template <template <typename> class T, typename... Args>
Kernel*
build_face_topo_kernel(stk::topology topo, Args&&... args)
{
  switch (topo.value()) {
  case stk::topology::QUAD_4:
    return new T<AlgTraitsQuad4>(std::forward<Args>(args)...);
  case stk::topology::TRI_3:
    return new T<AlgTraitsTri3>(std::forward<Args>(args)...);
  case stk::topology::LINE_2:
    return new T<AlgTraitsEdge_2D>(std::forward<Args>(args)...);
  default:
    return nullptr;
  }
}

template <template <typename> class T, typename... Args>
bool
build_topo_kernel_if_requested(
  stk::topology topo,
  EquationSystem& eqSys,
  std::vector<Kernel*>& kernelVec,
  std::string name,
  Args&&... args)
{
  bool isCreated = false;
  KernelBuilderLog::self().add_valid_name(eqSys.eqnTypeName_, name);
  if (eqSys.supp_alg_is_requested(name)) {
    Kernel* compKernel =
      build_topo_kernel<T>(topo, std::forward<Args>(args)...);
    STK_ThrowRequire(compKernel != nullptr);
    KernelBuilderLog::self().add_built_name(eqSys.eqnTypeName_, name);
    kernelVec.push_back(compKernel);
    isCreated = true;
  }
  return isCreated;
}

template <template <typename> class T, typename... Args>
bool
build_face_elem_topo_kernel_automatic(
  stk::topology faceTopo,
  stk::topology elemTopo,
  EquationSystem& eqSys,
  std::vector<Kernel*>& kernelVec,
  std::string name,
  Args&&... args)
{
  KernelBuilderLog::self().add_valid_name(eqSys.eqnTypeName_, name);
  Kernel* compKernel = build_face_elem_topo_kernel<T>(
    faceTopo, elemTopo, std::forward<Args>(args)...);
  STK_ThrowRequire(compKernel != nullptr);
  KernelBuilderLog::self().add_built_name(eqSys.eqnTypeName_, name);
  kernelVec.push_back(compKernel);
  return true;
}

template <template <typename> class T, typename... Args>
bool
build_fem_kernel_if_requested(
  stk::topology topo,
  EquationSystem& eqSys,
  std::vector<Kernel*>& kernelVec,
  std::string name,
  Args&&... args)
{
  bool isCreated = false;
  KernelBuilderLog::self().add_valid_name(eqSys.eqnTypeName_, name);
  if (eqSys.supp_alg_is_requested(name)) {
    Kernel* compKernel = build_fem_kernel<T>(topo, std::forward<Args>(args)...);
    STK_ThrowRequire(compKernel != nullptr);
    KernelBuilderLog::self().add_built_name(eqSys.eqnTypeName_, name);
    kernelVec.push_back(compKernel);
    isCreated = true;
  }
  return isCreated;
}

template <template <typename> class T, typename... Args>
bool
build_face_topo_kernel_automatic(
  stk::topology topo,
  EquationSystem& eqSys,
  std::vector<Kernel*>& kernelVec,
  std::string name,
  Args&&... args)
{
  KernelBuilderLog::self().add_valid_name(eqSys.eqnTypeName_, name);
  Kernel* compKernel =
    build_face_topo_kernel<T>(topo, std::forward<Args>(args)...);
  STK_ThrowRequire(compKernel != nullptr);
  KernelBuilderLog::self().add_built_name(eqSys.eqnTypeName_, name);
  kernelVec.push_back(compKernel);
  return true;
}

inline std::pair<AssembleFaceElemSolverAlgorithm*, bool>
build_or_add_part_to_face_elem_solver_alg(
  AlgorithmType /* algType */,
  EquationSystem& eqSys,
  stk::mesh::Part& part,
  stk::topology elemTopo,
  std::map<std::string, SolverAlgorithm*>& solverAlgs,
  const std::string bcName)
{
  const stk::topology topo = part.topology();
  const std::string algName = eqSys.name_ + "_" + bcName +
                              "_AssembleFaceElemSolverAlg_" + topo.name() +
                              "_" + elemTopo.name();

  bool isNotNGP =
    !(elemTopo == stk::topology::HEXAHEDRON_8 ||
      elemTopo == stk::topology::QUADRILATERAL_4_2D ||
      elemTopo == stk::topology::TRIANGLE_3_2D ||
      elemTopo == stk::topology::WEDGE_6 ||
      elemTopo == stk::topology::TETRAHEDRON_4 ||
      elemTopo == stk::topology::PYRAMID_5);
  STK_ThrowRequireMsg(
    !isNotNGP, "Consolidated algorithm called on non-NGP MasterElement");

  auto itc = solverAlgs.find(algName);
  bool createNewAlg = itc == solverAlgs.end();
  if (createNewAlg) {
    auto* theSolverAlg = new AssembleFaceElemSolverAlgorithm(
      eqSys.realm_, &part, &eqSys, topo.num_nodes(), elemTopo.num_nodes());
    STK_ThrowRequire(theSolverAlg != nullptr);

    NaluEnv::self().naluOutputP0()
      << "Created the following bc face/elem alg: " << algName << std::endl;
    solverAlgs.insert({algName, theSolverAlg});
  } else {
    auto& partVec = itc->second->partVec_;
    if (std::find(partVec.begin(), partVec.end(), &part) == partVec.end()) {
      partVec.push_back(&part);
    }
  }

  auto* theSolverAlg =
    dynamic_cast<AssembleFaceElemSolverAlgorithm*>(solverAlgs.at(algName));
  STK_ThrowRequire(theSolverAlg != nullptr);

  return {theSolverAlg, createNewAlg};
}

inline std::pair<AssembleElemSolverAlgorithm*, bool>
build_or_add_part_to_face_bc_solver_alg(
  EquationSystem& eqSys,
  stk::mesh::Part& part,
  std::map<std::string, SolverAlgorithm*>& solverAlgs,
  const std::string bcName)
{
  const stk::topology topo = part.topology();
  const std::string algName =
    eqSys.name_ + "_" + bcName + "_AssembleElemSolverAlg_" + topo.name();

  bool isNotNGP =
    !(topo == stk::topology::QUAD_4 || topo == stk::topology::QUAD_9 ||
      topo == stk::topology::TRI_3 || topo == stk::topology::LINE_2 ||
      topo == stk::topology::LINE_3);
  STK_ThrowRequireMsg(
    !isNotNGP, "Consolidated algorithm called on non-NGP MasterElement");

  auto itc = solverAlgs.find(algName);
  bool createNewAlg = itc == solverAlgs.end();
  if (createNewAlg) {
    auto* theSolverAlg = new AssembleElemSolverAlgorithm(
      eqSys.realm_, &part, &eqSys, eqSys.realm_.meta_data().side_rank(),
      topo.num_nodes());
    STK_ThrowRequire(theSolverAlg != nullptr);

    NaluEnv::self().naluOutputP0()
      << "Created the following bc face alg: " << algName << std::endl;
    solverAlgs.insert({algName, theSolverAlg});
  } else {
    auto& partVec = itc->second->partVec_;
    if (std::find(partVec.begin(), partVec.end(), &part) == partVec.end()) {
      partVec.push_back(&part);
    }
  }

  auto* theSolverAlg =
    dynamic_cast<AssembleElemSolverAlgorithm*>(solverAlgs.at(algName));
  STK_ThrowRequire(theSolverAlg != nullptr);

  return {theSolverAlg, createNewAlg};
}

} // namespace nalu
} // namespace sierra

#endif
