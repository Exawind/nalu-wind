// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <DgInfo.h>
#include <master_element/MasterElement.h>
#include <KynemaUGFEnv.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_topology/topology.hpp>

namespace sierra {
namespace kynema_ugf {

//==========================================================================
// Class Definition
//==========================================================================
// Acon_DgInfo - contains non-conformal DG-based information
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
DgInfo::DgInfo(
  int parallelRank,
  uint64_t globalFaceId,
  uint64_t localGaussPointId,
  int currentGaussPointId,
  stk::mesh::Entity currentFace,
  stk::mesh::Entity currentElement,
  const int currentFaceOrdinal,
  MasterElement* meFCCurrent,
  MasterElement* meSCSCurrent,
  stk::topology currentElementTopo,
  const int nDim,
  const double searchTolerance,
  MasterElement* meFCDevCurrent,
  MasterElement* meSCSDevCurrent)
  : parallelRank_(parallelRank),
    globalFaceId_(globalFaceId),
    localGaussPointId_(localGaussPointId),
    currentGaussPointId_(currentGaussPointId),
    currentFace_(currentFace),
    currentElement_(currentElement),
    currentFaceOrdinal_(currentFaceOrdinal),
    meFCCurrent_(meFCCurrent),
    meSCSCurrent_(meSCSCurrent),
    meFC_dev_Current_(meFCDevCurrent),
    meSCS_dev_Current_(meSCSDevCurrent),
    currentElementTopo_(currentElementTopo),
    nDim_(nDim),
    bestXRef_(1.0e16),
    bestX_(bestXRef_),
    nearestDistance_(searchTolerance),
    nearestDistanceSafety_(2.0),
    opposingFaceIsGhosted_(0)
{
  // resize internal vectors
  currentGaussPointCoords_.resize(nDim);
  // isoPar coords will map to full volume element
  currentIsoParCoords_.resize(nDim);
  opposingIsoParCoords_.resize(nDim);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
DgInfo::~DgInfo()
{
  // nothing to delete
}

//--------------------------------------------------------------------------
//-------- dump_info -------------------------------------------------------
//--------------------------------------------------------------------------
void
DgInfo::dump_info()
{
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "------------------------------------------------- " << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "DGInfo::dump_info() for localGaussPointId_ " << localGaussPointId_
    << " On Rank " << parallelRank_ << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "parallelRank_ " << parallelRank_ << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "globalFaceId_ " << globalFaceId_ << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "currentGaussPointId_ " << currentGaussPointId_ << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "currentFace_ " << currentFace_ << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "currentElement_ " << currentElement_ << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "currentElementTopo_ " << currentElementTopo_ << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput() << "nDim_ " << nDim_ << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "bestXRef_ " << bestXRef_ << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput() << "bestX_ " << bestX_ << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "nearestDistance_ " << nearestDistance_ << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "opposingFaceIsGhosted_ " << opposingFaceIsGhosted_ << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "opposingFace_ " << opposingFace_ << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput() << "opposingElement_ " << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "opposingElementTopo_ " << opposingElementTopo_ << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "opposingFaceOrdinal_ " << opposingFaceOrdinal_ << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "meFCOpposing_ " << meFCOpposing_ << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "meSCSOpposing_ " << meSCSOpposing_ << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "currentGaussPointCoords_ " << std::endl;
  for (size_t k = 0; k < currentGaussPointCoords_.size(); ++k)
    KynemaUGFEnv::self().kynema_ugfOutput()
      << currentGaussPointCoords_[k] << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "currentIsoParCoords_ " << std::endl;
  for (size_t k = 0; k < currentIsoParCoords_.size(); ++k)
    KynemaUGFEnv::self().kynema_ugfOutput()
      << currentIsoParCoords_[k] << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "opposingIsoParCoords_ " << std::endl;
  for (size_t k = 0; k < opposingIsoParCoords_.size(); ++k)
    KynemaUGFEnv::self().kynema_ugfOutput()
      << opposingIsoParCoords_[k] << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "allOpposingFaceIds_ " << std::endl;
  for (size_t k = 0; k < allOpposingFaceIds_.size(); ++k)
    KynemaUGFEnv::self().kynema_ugfOutput()
      << allOpposingFaceIds_[k] << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "allOpposingFaceIdsOld_ " << std::endl;
  for (size_t k = 0; k < allOpposingFaceIdsOld_.size(); ++k)
    KynemaUGFEnv::self().kynema_ugfOutput()
      << allOpposingFaceIdsOld_[k] << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput()
    << "------------------------------------------------- " << std::endl;
  KynemaUGFEnv::self().kynema_ugfOutput() << std::endl;
}

} // namespace kynema_ugf
} // namespace sierra
