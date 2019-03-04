/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


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
