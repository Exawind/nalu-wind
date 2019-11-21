// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef OversetInfo_h
#define OversetInfo_h

//==============================================================================
// Includes and forwards
//==============================================================================

#include <stk_mesh/base/Entity.hpp>
#include <cmath> 
#include <vector>

namespace sierra {
namespace nalu {

class MasterElement;

//=============================================================================
// Class Definition
//=============================================================================
// OversetInfo
//=============================================================================
class OversetInfo {

 public:

  // constructor and destructor
  OversetInfo(
    stk::mesh::Entity node,
    const int nDim );

  ~OversetInfo();

  stk::mesh::Entity orphanNode_;
  stk::mesh::Entity owningElement_;

  double bestX_;
  int elemIsGhosted_;

  // master element for background mesh
  MasterElement *meSCS_;

  std::vector<double> isoParCoords_;
  std::vector<double> nodalCoords_;

};
  
} // end sierra namespace
} // end nalu namespace

#endif
