// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AlgTraits_h
#define AlgTraits_h

#include <stk_topology/topology.hpp>

namespace sierra {
namespace nalu {

class HexSCS;
class HexSCV;
class TetSCS;
class TetSCV;
class PyrSCS;
class PyrSCV;
class WedSCS;
class WedSCV;
class Quad42DSCS;
class Quad42DSCV;
class Quad3DSCS;
class Tri32DSCS;
class Tri32DSCV;
class Tri3DSCS;
class Edge2DSCS;

// limited supported now (P=1 3D elements)
struct AlgTraitsHex8
{
  static constexpr int nDim_ = 3;
  static constexpr int nodesPerElement_ = 8;
  static constexpr int numScsIp_ = 12;
  static constexpr int numScvIp_ = 8;
  static constexpr int numGp_ = 8; // for FEM
  static constexpr stk::topology::topology_t topo_ = stk::topology::HEX_8;
  using masterElementScs_ = HexSCS;
  using masterElementScv_ = HexSCV;
};

struct AlgTraitsTet4
{
  static constexpr int nDim_ = 3;
  static constexpr int nodesPerElement_ = 4;
  static constexpr int numScsIp_ = 6;
  static constexpr int numScvIp_ = 4;
  static constexpr int numGp_ = 4; // for FEM (not supported)
  static constexpr stk::topology::topology_t topo_ = stk::topology::TET_4;
  using masterElementScs_ = TetSCS;
  using masterElementScv_ = TetSCV;
};

struct AlgTraitsPyr5
{
  static constexpr int nDim_ = 3;
  static constexpr int nodesPerElement_ = 5;
  static constexpr int numScsIp_ = 12;
  static constexpr int numScvIp_ = 5;
  static constexpr int numGp_ = 5; // for FEM (not supported)
  static constexpr stk::topology::topology_t topo_ = stk::topology::PYRAMID_5;
  using masterElementScs_ = PyrSCS;
  using masterElementScv_ = PyrSCV;
};

struct AlgTraitsWed6
{
  static constexpr int nDim_ = 3;
  static constexpr int nodesPerElement_ = 6;
  static constexpr int numScsIp_ = 9;
  static constexpr int numScvIp_ = 6;
  static constexpr int numGp_ = 6; // for FEM (not supported)
  static constexpr stk::topology::topology_t topo_ = stk::topology::WEDGE_6;
  using masterElementScs_ = WedSCS;
  using masterElementScv_ = WedSCV;
};

struct AlgTraitsQuad4_2D
{
  static constexpr int nDim_ = 2;
  static constexpr int nodesPerElement_ = 4;
  static constexpr int numScsIp_ = 4;
  static constexpr int numScvIp_ = 4;
  static constexpr int numGp_ = 4; // for FEM (not supported)
  static constexpr stk::topology::topology_t topo_ = stk::topology::QUAD_4_2D;
  using masterElementScs_ = Quad42DSCS;
  using masterElementScv_ = Quad42DSCV;
};

struct AlgTraitsTri3_2D
{
  static constexpr int nDim_ = 2;
  static constexpr int nodesPerElement_ = 3;
  static constexpr int numScsIp_ = 3;
  static constexpr int numScvIp_ = 3;
  static constexpr int numGp_ = 3; // for FEM (not supported)
  static constexpr stk::topology::topology_t topo_ = stk::topology::TRI_3_2D;
  using masterElementScs_ = Tri32DSCS;
  using masterElementScv_ = Tri32DSCV;
};

struct AlgTraitsEdge_3D
{
  static constexpr int nDim_ = 3;
  static constexpr int nodesPerElement_ = 2;
  static constexpr int numScsIp_ = 1;
  static constexpr int numScvIp_ = 2;
  static constexpr stk::topology::topology_t topo_ = stk::topology::LINE_2;
};

//-------------------------------------------------------------------------------------------

struct AlgTraitsQuad4
{
  static constexpr int nDim_ = 3;
  static constexpr int nodesPerElement_ = 4;
  static constexpr int nodesPerFace_ = nodesPerElement_;
  static constexpr int numScsIp_ = 4;
  static constexpr int numFaceIp_ = numScsIp_;
  static constexpr stk::topology::topology_t topo_ = stk::topology::QUAD_4;
  using masterElementScs_ = Quad3DSCS;
};

struct AlgTraitsShellQuad4 : public AlgTraitsQuad4
{
  static constexpr stk::topology::topology_t topo_ =
    stk::topology::SHELL_QUAD_4;
};

struct AlgTraitsTri3
{
  static constexpr int nDim_ = 3;
  static constexpr int nodesPerElement_ = 3;
  static constexpr int nodesPerFace_ = nodesPerElement_;
  static constexpr int numScsIp_ = 3;
  static constexpr int numFaceIp_ = numScsIp_;
  static constexpr stk::topology::topology_t topo_ = stk::topology::TRI_3;
  using masterElementScs_ = Tri3DSCS;
};

struct AlgTraitsShellTri3 : public AlgTraitsTri3
{
  static constexpr stk::topology::topology_t topo_ = stk::topology::SHELL_TRI_3;
};

struct AlgTraitsEdge_2D
{
  static constexpr int nDim_ = 2;
  static constexpr int nodesPerElement_ = 2;
  static constexpr int nodesPerFace_ = nodesPerElement_;
  static constexpr int numScsIp_ = 2;
  static constexpr int numFaceIp_ = numScsIp_;
  static constexpr stk::topology::topology_t topo_ = stk::topology::LINE_2;
  using masterElementScs_ = Edge2DSCS;
};

struct AlgTraitsBeam_2D : public AlgTraitsEdge_2D
{
  static constexpr stk::topology::topology_t topo_ = stk::topology::BEAM_2;
};
//-------------------------------------------------------------------------------------------

template <typename AlgTraitsFace, typename AlgTraitsElem>
struct AlgTraitsFaceElem
{
  using FaceTraits = AlgTraitsFace;
  using ElemTraits = AlgTraitsElem;

  static constexpr int nDim_ = ElemTraits::nDim_;
  static_assert(
    nDim_ == FaceTraits::nDim_, "inconsistent dimension specification");

  static constexpr int nodesPerElement_ = ElemTraits::nodesPerElement_;
  static constexpr int nodesPerFace_ = FaceTraits::nodesPerElement_;

  static constexpr int numScsIp_ = ElemTraits::numScsIp_;
  static constexpr int numScvIp_ = ElemTraits::numScvIp_;

  static constexpr int numFaceIp_ = FaceTraits::numScsIp_;

  static constexpr stk::topology::topology_t elemTopo_ = ElemTraits::topo_;
  static constexpr stk::topology::topology_t faceTopo_ = FaceTraits::topo_;
};

using AlgTraitsEdge2DTri32D =
  AlgTraitsFaceElem<AlgTraitsEdge_2D, AlgTraitsTri3_2D>;
using AlgTraitsEdge2DQuad42D =
  AlgTraitsFaceElem<AlgTraitsEdge_2D, AlgTraitsQuad4_2D>;

using AlgTraitsTri3Tet4 = AlgTraitsFaceElem<AlgTraitsTri3, AlgTraitsTet4>;
using AlgTraitsTri3Pyr5 = AlgTraitsFaceElem<AlgTraitsTri3, AlgTraitsPyr5>;
using AlgTraitsTri3Wed6 = AlgTraitsFaceElem<AlgTraitsTri3, AlgTraitsWed6>;

using AlgTraitsQuad4Hex8 = AlgTraitsFaceElem<AlgTraitsQuad4, AlgTraitsHex8>;
using AlgTraitsQuad4Pyr5 = AlgTraitsFaceElem<AlgTraitsQuad4, AlgTraitsPyr5>;
using AlgTraitsQuad4Wed6 = AlgTraitsFaceElem<AlgTraitsQuad4, AlgTraitsWed6>;

} // namespace nalu
} // namespace sierra

#endif
