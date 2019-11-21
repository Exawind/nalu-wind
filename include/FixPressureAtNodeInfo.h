// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef FIXPRESSUREATNODEINFO_H
#define FIXPRESSUREATNODEINFO_H

#include "stk_mesh/base/MetaData.hpp"
#include "stk_search/SearchMethod.hpp"

#include <vector>
#include <string>

namespace sierra {
namespace nalu {

/** Input data for fixing pressure during ABL simulations.
 *
 *  This class holds the user data parsed from Nalu input file to fix pressure
 *  at a spatial location for ABL simulations. This is especially necessary for
 *  precursor simulations where the inlet/outlet boundaries are periodic and the
 *  top boundary is symmetric. In this case, there is no pressure specified for
 *  the continuity solve.
 */
struct FixPressureAtNodeInfo
{
  /** Options for specifying the node where pressure is fixed
   *
   */
  enum NodeLookupType {
    STK_NODE_ID = 0,      //!< STK Global Node ID provided in input file
    SPATIAL_LOCATION = 1, //!< Spatial location used to search nearest node
  };

  void create_part_vector(const stk::mesh::MetaData& meta)
  {
    auto nParts = searchParts_.size();
    for (size_t i=0; i < nParts; i++) {
      stk::mesh::Part* part = meta.get_part(searchParts_[i]);
      if (nullptr != part)
        partVec_.push_back(part);
      else
        throw std::runtime_error("FixPressureAtNodeInfo: Target search part is null "
                                 + searchParts_[i]);
    }
  }

  //! Reference pressure for ABL simulations
  double refPressure_{0.0};

  //! Location for ABL reference pressure
  std::vector<double> location_{0.0, 0.0, 0.0};

  std::vector<std::string> searchParts_;

  stk::mesh::PartVector partVec_;

  //! Search method for determining the nearest node to the prescribed location
  stk::search::SearchMethod searchMethod_{stk::search::KDTREE};

  NodeLookupType lookupType_{SPATIAL_LOCATION};

  stk::mesh::EntityId stkNodeId_;
};

}  // nalu
}  // sierra

#endif /* FIXPRESSUREATNODEINFO_H */
