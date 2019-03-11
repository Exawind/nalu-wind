/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef INTERFACEBALANCER_H_
#define INTERFACEBALANCER_H_

#include <set>
#include <map>
#include <vector>

#include "stk_mesh/base/Types.hpp"

namespace stk { namespace mesh { class BulkData; } }

namespace sierra {
namespace nalu {

class InterfaceBalancer {

public:
  InterfaceBalancer(const stk::mesh::MetaData& meta,
                    stk::mesh::BulkData& bulk);

 void balance_node_entities(const double targetLoadBalance,
                            const int maxIterations);

private:

  void  getInterfaceDescription (std::set<int>& neighborProcessors,
                                 std::map<stk::mesh::Entity, std::vector<int> >& interfaceNodesAndProcessors);

  void getGlobalLoadImbalance (double &loadFactor,  int& numLocallyOwnedNodes);

  void exchangeLocalSizes (const std::set<int>& neighborProcessors,
                           int& numLocallyOwnedNodes,  std::map<int, int>& numLocallyOwnedByRank);
  void
  changeOwnersOfNodes (const std::map<stk::mesh::Entity, std::vector<int> >& interfaceNodesAndProcessors,
      std::map<int, int>& numLocallyOwnedByRank,
      int numLocallyOwnedNodes);

  const stk::mesh::MetaData & metaData_;
  stk::mesh::BulkData & bulkData_;
//  const double tolerance_;
};
}
}

#endif /* INTERFACEBALANCER_H_ */
