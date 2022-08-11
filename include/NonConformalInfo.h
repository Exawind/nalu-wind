// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NonConformalInfo_h
#define NonConformalInfo_h

//==============================================================================
// Includes and forwards
//==============================================================================

#include <master_element/MasterElement.h>

// stk
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Ghosting.hpp>

#include <stk_search/BoundingBox.hpp>
#include <stk_search/IdentProc.hpp>
#include <stk_search/SearchMethod.hpp>

#include <vector>
#include <map>

namespace stk {
namespace mesh {
typedef std::vector<Part*> PartVector;
}
} // namespace stk

namespace sierra {
namespace nalu {

class Realm;
class DgInfo;

typedef stk::search::IdentProc<uint64_t, int> theKey;
typedef stk::search::Point<double> Point;
typedef stk::search::Sphere<double> Sphere;
typedef stk::search::Box<double> Box;
typedef std::pair<Sphere, theKey> boundingSphere;
typedef std::pair<Box, theKey> boundingElementBox;

//=============================================================================
// Class Definition
//=============================================================================
// NonConformalInfo
//=============================================================================
/**
 * * @par Description:
 * - class to manage dg information.
 *
 * @par Design Considerations:
 * -
 */
//=============================================================================
class NonConformalInfo
{

public:
  // constructor and destructor
  NonConformalInfo(
    Realm& realm,
    const stk::mesh::PartVector currentPartVec,
    const stk::mesh::PartVector opposingPartVec,
    const double expandBoxPercentage,
    const std::string& searchMethodName,
    const bool clipIsoParametricCoords,
    const double searchTolerance,
    const bool dynamicSearchTolAlg,
    const std::string debugName);

  ~NonConformalInfo();

  /* delete dgInfoVec_ */
  void delete_dgInfo();

  /* perform initialization such as dgInfoVec creation and search point/boxes */
  void initialize();

  /* perform the 'new' */
  void construct_dgInfo();

  void reset_dgInfo();
  void construct_bounding_points();
  void construct_bounding_boxes();
  void determine_elems_to_ghost();
  void complete_search();
  void provide_diagnosis();
  size_t error_check();

  Realm& realm_;
  const std::string name_;

  // master slave parts; slave part can be subsetted while master is not..
  const stk::mesh::PartVector currentPartVec_;
  const stk::mesh::PartVector opposingPartVec_;

  /* expand search box */
  double expandBoxPercentage_;

  stk::search::SearchMethod searchMethod_;

  /* clip isoparametric coordinates if they are out of bounds */
  const bool clipIsoParametricCoords_;

  /* allow for some finite search tolereance for bounding box */
  const double searchTolerance_;

  /* allow for dynamic search tolerance algorithm where search tolerance is used
   * as point radius from isInElem */
  const bool dynamicSearchTolAlg_;

  /* does the realm have mesh motion */
  const bool meshMotion_;

  /* can we possibly reuse */
  bool canReuse_;

  /* bounding box data types for stk_search */
  std::vector<boundingSphere> boundingSphereVec_;
  std::vector<boundingElementBox> boundingFaceElementBoxVec_;

  /* vector of DgInfo */
  std::vector<std::vector<DgInfo*>> dgInfoVec_;

  /* save off product of search */
  std::vector<std::pair<theKey, theKey>> searchKeyPair_;

private:
  void delete_range_points_found(
    std::vector<boundingSphere>& boundingSphereVec,
    const std::vector<std::pair<theKey, theKey>>& searchKeyPair) const;
  void repeat_search_if_needed(
    const std::vector<boundingSphere>& boundingSphereVec,
    std::vector<std::pair<theKey, theKey>>& searchKeyPair) const;
};

} // namespace nalu
} // namespace sierra

#endif
