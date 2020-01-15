// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef NonConformalManager_h
#define NonConformalManager_h

//==============================================================================
// Includes and forwards
//==============================================================================

// stk
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Ghosting.hpp>

#include <vector>
#include <map>

namespace sierra {
namespace nalu {

class DgInfo;
class Realm;
class NonConformalInfo;

//=============================================================================
// Class Definition
//=============================================================================
// NonConformalManager
//=============================================================================
/**
 * * @par Description:
 * - class to manage all NonConformalInfo objects.
 *
 * @par Design Considerations:
 * -
 */
//=============================================================================
class NonConformalManager {

 public:

  // constructor and destructor
  NonConformalManager(
    Realm & realm,
    const bool ncAlgDetailedOutput,
    const bool ncAlgCoincidentNodesErrorCheck);

  ~NonConformalManager();

  void initialize();

  Realm &realm_;
  const bool ncAlgDetailedOutput_;
  const bool ncAlgCoincidentNodesErrorCheck_;

  /* ghosting for all surface:block pair */
  stk::mesh::Ghosting *nonConformalGhosting_;

  stk::mesh::EntityProcVec elemsToGhost_;
  std::vector<NonConformalInfo *> nonConformalInfoVec_;

  std::vector<int> ghostCommProcs_;

  private:

  void manage_ghosting(std::vector<stk::mesh::EntityKey>& recvGhostsToRemove);
};

} // end nalu namespace
} // end sierra namespace

#endif
