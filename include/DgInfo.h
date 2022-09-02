// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef DgInfo_h
#define DgInfo_h

//==============================================================================
// Includes and forwards
//==============================================================================

#include <stk_mesh/base/Entity.hpp>
#include <stk_topology/topology.hpp>

#include <vector>

namespace sierra {
namespace nalu {

class MasterElement;

//=============================================================================
// Class Definition
//=============================================================================
// DgInfo
//=============================================================================
/**
 * * @par Description:
 * - class for halo stuff.
 *
 * @par Design Considerations:
 * -
 */
//=============================================================================
class DgInfo
{

public:
  // constructor and destructor
  DgInfo(
    int parallelRank,
    uint64_t globalFaceId,
    uint64_t localGaussPointId,
    int currentGaussPointId,
    stk::mesh::Entity currentFace,
    stk::mesh::Entity currentElement,
    const int currentFaceOrdinal,
    MasterElement* meFCCurrent,
    MasterElement* meSCSurrent,
    stk::topology currentElementTopo,
    const int nDim,
    double searchTolerance,
    MasterElement* meFCDevCurrent = nullptr,
    MasterElement* meSCSDevCurrent = nullptr);

  ~DgInfo();

  void dump_info();

  int parallelRank_;
  uint64_t globalFaceId_;
  uint64_t localGaussPointId_;
  int currentGaussPointId_;

  stk::mesh::Entity currentFace_;
  stk::mesh::Entity currentElement_;
  const int currentFaceOrdinal_;
  MasterElement* meFCCurrent_;
  MasterElement* meSCSCurrent_;
  MasterElement* meFC_dev_Current_;
  MasterElement* meSCS_dev_Current_;
  stk::topology currentElementTopo_;

  int nDim_;

  const double bestXRef_;
  double bestX_;
  double nearestDistance_;
  const double nearestDistanceSafety_;

  int opposingFaceIsGhosted_;

  // search provides opposing face
  stk::mesh::Entity opposingFace_;

  // face:element relations provide connected element to opposing face
  stk::mesh::Entity opposingElement_;

  // opposing element topo
  stk::topology opposingElementTopo_;

  // for the opposing face, what is its ordinal?
  int opposingFaceOrdinal_;

  // master element for opposing face
  MasterElement* meFCOpposing_;

  // master element for opposing face connected element
  MasterElement* meSCSOpposing_;

  // coordinates of gauss points on current face
  std::vector<double> currentGaussPointCoords_;

  // iso-parametric coordinates for gauss point on current face (-1:1)
  std::vector<double> currentIsoParCoords_;

  // iso-parametric coordinates for gauss point on opposing face (-1:1)
  std::vector<double> opposingIsoParCoords_;

  // possible reuse
  std::vector<uint64_t> allOpposingFaceIds_;
  std::vector<uint64_t> allOpposingFaceIdsOld_;
};

} // namespace nalu
} // namespace sierra

#endif
