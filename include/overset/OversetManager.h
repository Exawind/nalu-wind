/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef OVERSETMANAGER_H
#define OVERSETMANAGER_H

#include <stk_mesh/base/Selector.hpp>

#include <vector>

namespace stk {
namespace io {
class StkMeshIoBroker;
}

namespace mesh {
class Part;
class MetaData;
class BulkData;
class Ghosting;
class FieldBase;
typedef std::vector<Part*> PartVector;
struct Entity;
}
}

namespace sierra {
namespace nalu {

class Realm;
class OversetInfo;

/** Base class for Overset connectivity manager
 *
 */
class OversetManager
{
public:
  OversetManager(Realm& realm);

  virtual ~OversetManager();

  /** Deallocate OversetInfo memory allocated via new upoin reinitialization */
  void delete_info_vec();

  /** Perform any necessary setup actions for overset algorithms
   *
   *  Part and field registration to STK
   */
  virtual void setup();

  /** Setup all data structures and perform connectivity
   *
   * This method must be implemented by concrete OGA implementations.
   */
  virtual void initialize() = 0;

  virtual void overset_orphan_node_field_update(
    stk::mesh::FieldBase*,
    const int,
    const int);

  /** Return an inactive selector that contains the hole elements
   */
  virtual stk::mesh::Selector get_inactive_selector();

  Realm& realm_;

  stk::mesh::MetaData* metaData_{nullptr};

  stk::mesh::BulkData* bulkData_{nullptr};

  stk::mesh::Ghosting* oversetGhosting_{nullptr};

  stk::mesh::Part* inActivePart_{nullptr};

  stk::mesh::Part* backgroundSurfacePart_{nullptr};

  stk::mesh::PartVector orphanPointSurfaceVecBackground_;

  std::vector<OversetInfo*> oversetInfoVec_;

  std::vector<stk::mesh::Entity> holeNodes_;

  std::vector<int> ghostCommProcs_;

private:
  OversetManager() = delete;
  OversetManager(const OversetManager&) = delete;

};

} // nalu
} // sierra

#endif /* OVERSETMANAGER_H */
