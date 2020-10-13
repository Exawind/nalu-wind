// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef PeriodicManager_h
#define PeriodicManager_h

//==============================================================================
// Includes and forwards
//==============================================================================

#include <FieldTypeDef.h>
#include <KokkosInterface.h>

// stk
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Ghosting.hpp>

#include <stk_search/BoundingBox.hpp>
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>

#include <vector>
#include <list>
#include <map>

namespace sierra {
namespace nalu {

class Realm;

typedef stk::search::IdentProc<stk::mesh::EntityKey,int> theEntityKey;
typedef stk::search::Point<double> Point;
typedef stk::search::Sphere<double> Sphere;
typedef std::pair<Sphere,theEntityKey> sphereBoundingBox;

class PeriodicManager {

 public:

  // constructor and destructor
  PeriodicManager(
    Realm & realm);

  ~PeriodicManager();

  void initialize_error_count();

  void add_periodic_pair(
    stk::mesh::Part* meshPartsMaster,
    stk::mesh::Part* meshPartsSlave,
    const double &userSearchTolerance,
    const std::string &searchMethodName);

  void build_constraints();

  // holder for master += slave; slave = master
  void apply_constraints(
    stk::mesh::FieldBase *,
    const unsigned &sizeOfField,
    const bool &bypassFieldCheck,
    const bool &addSlaves = true,
    const bool &setSlaves = true);

  void ngp_apply_constraints(
    stk::mesh::FieldBase *,
    const unsigned &sizeOfField,
    const bool &bypassFieldCheck,
    const bool &addSlaves = true,
    const bool &setSlaves = true,
    const bool &doCommunication = true) const;

  // find the max
  void apply_max_field(
    stk::mesh::FieldBase *,
    const unsigned &sizeOfField);

  void manage_ghosting_object();

  stk::mesh::Ghosting * get_ghosting_object();

  const stk::mesh::PartVector &get_slave_part_vector();

  const stk::mesh::PartVector& periodic_parts_vector()
  {
    return periodicPartVec_;
  }

  double get_search_time();

  void augment_periodic_selector_pairs();

  void initialize_translation_vector();

  void determine_translation(
     stk::mesh::Selector masterSelector,
     stk::mesh::Selector slaveSelector,
     std::vector<double> &translationVector,
     std::vector<double> &rotationVector);

  void remove_redundant_slave_nodes();

  void finalize_search();

  void populate_search_key_vec(
    stk::mesh::Selector masterSelector,
    stk::mesh::Selector slaveSelector,
    std::vector<double> &translationVector,
    const stk::search::SearchMethod searchMethod);

  void error_check();

  void update_global_id_field();

  /* communicate periodicGhosting nodes */
  void
  periodic_parallel_communicate_field(
    stk::mesh::FieldBase *theField);

  void
  ngp_periodic_parallel_communicate_field(
    stk::mesh::FieldBase *theField) const;

  /* communicate shared nodes and aura nodes */
  void
  parallel_communicate_field(
    stk::mesh::FieldBase *theField);

  void
  ngp_parallel_communicate_field(
    stk::mesh::FieldBase *theField) const;

  Realm &realm_;

  /* manage tolerances; each block specifies a user tolerance */
  double searchTolerance_;

  stk::mesh::Ghosting *periodicGhosting_;
  const std::string ghostingName_;
  double timerSearch_;

  // algorithm to find/exclude points when M/S does not match
  int errorCount_;
  int maxErrorCount_;
  const double amplificationFactor_;

 public:
  // the data structures to hold master/slave information
  typedef std::pair<stk::mesh::Entity, stk::mesh::Entity> EntityPair;
  typedef Kokkos::pair<stk::mesh::Entity, stk::mesh::Entity> KokkosEntityPair;
  typedef std::pair<stk::mesh::Selector, stk::mesh::Selector> SelectorPair;
  typedef std::vector<std::pair<theEntityKey,theEntityKey> > SearchKeyVector;
  typedef Kokkos::View<KokkosEntityPair*, Kokkos::LayoutRight, MemSpace> KokkosEntityPairView;

  std::vector<int> ghostCommProcs_;

  void ngp_add_slave_to_master(
    stk::mesh::FieldBase *theField,
    const unsigned &sizeOfField,
    const bool &bypassFieldCheck,
    const bool &doCommunication) const;

  void ngp_set_slave_to_master(
    stk::mesh::FieldBase *theField,
    const unsigned &sizeOfField,
    const bool &bypassFieldCheck,
    const bool &doCommunication) const;

 private:

  // vector of master:slave selector pairs
  std::vector<SelectorPair> periodicSelectorPairs_;

  stk::mesh::PartVector periodicPartVec_;

  // vector of slave parts
  stk::mesh::PartVector slavePartVector_;

  // vector of search types
  std::vector<stk::search::SearchMethod> searchMethodVec_;

  // translation and rotation
  std::vector<std::vector<double> > translationVector_;
  std::vector<std::vector<double> > rotationVector_;

  // vector of masterEntity:slaveEntity
  std::vector<EntityPair> masterSlaveCommunicator_;
  KokkosEntityPairView deviceMasterSlaves_;
  KokkosEntityPairView::HostMirror hostMasterSlaves_;

  // culmination of all searches
  SearchKeyVector searchKeyVector_;

  void add_slave_to_master(
    stk::mesh::FieldBase *theField,
    const unsigned &sizeOfField,
    const bool &bypassFieldCheck);

  void set_slave_to_master(
    stk::mesh::FieldBase *theField,
    const unsigned &sizeOfField,
    const bool &bypassFieldCheck);

};

} // namespace nalu
} // namespace sierra

#endif
