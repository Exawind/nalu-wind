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
#include <NaluEnv.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_topology/topology.hpp>

namespace sierra {
namespace nalu {

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
  const double searchTolerance)
  : parallelRank_(parallelRank),
    globalFaceId_(globalFaceId),
    localGaussPointId_(localGaussPointId),
    currentGaussPointId_(currentGaussPointId),
    currentFace_(currentFace),
    currentElement_(currentElement),
    currentFaceOrdinal_(currentFaceOrdinal),
    meFCCurrent_(meFCCurrent),
    meSCSCurrent_(meSCSCurrent),
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
  NaluEnv::self().naluOutput()
    << "------------------------------------------------- " << std::endl;
  NaluEnv::self().naluOutput()
    << "DGInfo::dump_info() for localGaussPointId_ " << localGaussPointId_
    << " On Rank " << parallelRank_ << std::endl;
  NaluEnv::self().naluOutput()
    << "parallelRank_ " << parallelRank_ << std::endl;
  NaluEnv::self().naluOutput()
    << "globalFaceId_ " << globalFaceId_ << std::endl;
  NaluEnv::self().naluOutput()
    << "currentGaussPointId_ " << currentGaussPointId_ << std::endl;
  NaluEnv::self().naluOutput() << "currentFace_ " << currentFace_ << std::endl;
  NaluEnv::self().naluOutput()
    << "currentElement_ " << currentElement_ << std::endl;
  NaluEnv::self().naluOutput()
    << "currentElementTopo_ " << currentElementTopo_ << std::endl;
  NaluEnv::self().naluOutput() << "nDim_ " << nDim_ << std::endl;
  NaluEnv::self().naluOutput() << "bestXRef_ " << bestXRef_ << std::endl;
  NaluEnv::self().naluOutput() << "bestX_ " << bestX_ << std::endl;
  NaluEnv::self().naluOutput()
    << "nearestDistance_ " << nearestDistance_ << std::endl;
  NaluEnv::self().naluOutput()
    << "opposingFaceIsGhosted_ " << opposingFaceIsGhosted_ << std::endl;
  NaluEnv::self().naluOutput()
    << "opposingFace_ " << opposingFace_ << std::endl;
  NaluEnv::self().naluOutput() << "opposingElement_ " << std::endl;
  NaluEnv::self().naluOutput()
    << "opposingElementTopo_ " << opposingElementTopo_ << std::endl;
  NaluEnv::self().naluOutput()
    << "opposingFaceOrdinal_ " << opposingFaceOrdinal_ << std::endl;
  NaluEnv::self().naluOutput()
    << "meFCOpposing_ " << meFCOpposing_ << std::endl;
  NaluEnv::self().naluOutput()
    << "meSCSOpposing_ " << meSCSOpposing_ << std::endl;
  NaluEnv::self().naluOutput() << "currentGaussPointCoords_ " << std::endl;
  for (size_t k = 0; k < currentGaussPointCoords_.size(); ++k)
    NaluEnv::self().naluOutput() << currentGaussPointCoords_[k] << std::endl;
  NaluEnv::self().naluOutput() << "currentIsoParCoords_ " << std::endl;
  for (size_t k = 0; k < currentIsoParCoords_.size(); ++k)
    NaluEnv::self().naluOutput() << currentIsoParCoords_[k] << std::endl;
  NaluEnv::self().naluOutput() << "opposingIsoParCoords_ " << std::endl;
  for (size_t k = 0; k < opposingIsoParCoords_.size(); ++k)
    NaluEnv::self().naluOutput() << opposingIsoParCoords_[k] << std::endl;
  NaluEnv::self().naluOutput() << "allOpposingFaceIds_ " << std::endl;
  for (size_t k = 0; k < allOpposingFaceIds_.size(); ++k)
    NaluEnv::self().naluOutput() << allOpposingFaceIds_[k] << std::endl;
  NaluEnv::self().naluOutput() << "allOpposingFaceIdsOld_ " << std::endl;
  for (size_t k = 0; k < allOpposingFaceIdsOld_.size(); ++k)
    NaluEnv::self().naluOutput() << allOpposingFaceIdsOld_[k] << std::endl;
  NaluEnv::self().naluOutput()
    << "------------------------------------------------- " << std::endl;
  NaluEnv::self().naluOutput() << std::endl;
}

} // namespace nalu
} // namespace sierra
